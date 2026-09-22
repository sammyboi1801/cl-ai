# cl-ai PowerShell integration.
#
# Type what you want, press Tab, cycle with the arrow keys, and the command
# lands in your prompt ready to read and run. Nothing is ever executed for you.
#
# This module is deliberately thin. It holds exactly one piece of state -- which
# candidate is highlighted -- and every decision lives in the daemon, so this
# file is written once and rarely touched. It is the layer we least want to
# debug, because a bug here breaks the user's terminal rather than one feature.
#
# The rule that governs everything below: if anything at all goes wrong -- no
# daemon, a slow daemon, a version mismatch, a garbled reply -- Tab must behave
# exactly like ordinary Tab. A completion that hangs a prompt is far worse than
# one that never appears, so every failure path falls through to the built-in
# handler rather than surfacing an error.

Set-StrictMode -Version Latest

$script:Candidates   = @()
$script:Index        = -1
$script:OriginalLine = $null
$script:ProtocolVersion = 1
$script:TimeoutMs    = 500

function Get-ClAiPortFile {
    <#
        .SYNOPSIS
        Locate the file holding the daemon's loopback port.

        .DESCRIPTION
        Mirrors port_file_for() in transport.py. The Windows transport is
        currently a loopback stand-in for a named pipe; when pipes land, this
        function and Invoke-ClAiRequest are the only two that change.
    #>
    [CmdletBinding()]
    param([string]$Endpoint)

    # An explicit override wins. This exists so a user can point the widget at
    # a non-default daemon -- a second instance, or one started by hand for
    # debugging -- and it is also how the test suite drives this module without
    # touching the real per-user endpoint.
    if ($env:CL_AI_PORT_FILE) { return $env:CL_AI_PORT_FILE }
    if ($env:CL_AI_ENDPOINT -and -not $Endpoint) { $Endpoint = $env:CL_AI_ENDPOINT }

    if (-not $Endpoint) {
        $user = $env:USERNAME
        if (-not $user) { $user = 'default' }
        $safeUser = ($user.ToCharArray() | Where-Object { $_ -match '[\w-]' }) -join ''
        $Endpoint = "\\.\pipe\cl-ai-$safeUser"
    }
    $safe = ($Endpoint.ToCharArray() | Where-Object { $_ -match '[\w-]' }) -join ''
    Join-Path $env:TEMP "$safe.port"
}

function Invoke-ClAiRequest {
    <#
        .SYNOPSIS
        Send one request to the daemon. Returns $null on any failure.

        .DESCRIPTION
        $null is the whole error vocabulary on purpose: the widget cannot
        usefully tell "no daemon" from "too slow" from "bad reply", and the
        response to all three is identical -- let Tab do its normal job.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Buffer,
        [int]$Cursor = 0,
        [int]$Limit  = 5
    )

    $client = $null
    try {
        $portFile = Get-ClAiPortFile
        if (-not (Test-Path -LiteralPath $portFile)) { return $null }
        $port = [int](Get-Content -LiteralPath $portFile -Raw).Trim()

        $request = [ordered]@{
            kind        = 'suggest'
            buffer      = $Buffer
            cursor      = $Cursor
            shell       = 'powershell'
            cwd         = (Get-Location).Path
            limit       = $Limit
            deadline_ms = $script:TimeoutMs
            version     = $script:ProtocolVersion
        }
        $json = ($request | ConvertTo-Json -Compress -Depth 4)

        $client = New-Object System.Net.Sockets.TcpClient
        # Connect with a timeout: TcpClient.Connect() blocks indefinitely by
        # default, which at a prompt means a frozen terminal.
        $connect = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne($script:TimeoutMs)) { return $null }
        $client.EndConnect($connect)

        $client.SendTimeout    = $script:TimeoutMs
        $client.ReceiveTimeout = $script:TimeoutMs
        $stream = $client.GetStream()

        $payload = [System.Text.Encoding]::UTF8.GetBytes($json + "`n")
        $stream.Write($payload, 0, $payload.Length)
        $stream.Flush()

        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        $line = $reader.ReadLine()
        if (-not $line) { return $null }
        return ($line | ConvertFrom-Json)
    }
    catch {
        # Intentionally silent. An error written here would corrupt the prompt
        # the user is in the middle of typing.
        return $null
    }
    finally {
        if ($client) { $client.Dispose() }
    }
}

function Get-ClAiSuggestions {
    <#
        .SYNOPSIS
        Ask the daemon for candidates. Always returns an array, never $null.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][AllowEmptyString()][string]$Buffer)

    if ([string]::IsNullOrWhiteSpace($Buffer)) { return @() }

    $response = Invoke-ClAiRequest -Buffer $Buffer
    if (-not $response) { return @() }

    # A version mismatch is a normal outcome, not a crash: the user upgraded
    # the package without restarting this shell. Say so once, quietly.
    if ($response.PSObject.Properties.Name -contains 'error' -and
        $response.error -eq 'version_mismatch') {
        Write-Host ''
        Write-Host "cl-ai: $($response.message)" -ForegroundColor DarkYellow
        return @()
    }
    if (-not $response.ok) { return @() }
    if (-not $response.suggestions) { return @() }
    return @($response.suggestions)
}

function Reset-ClAiCycle {
    $script:Candidates   = @()
    $script:Index        = -1
    $script:OriginalLine = $null
}

function Show-ClAiCandidate {
    <#
        .SYNOPSIS
        Replace the edit buffer with the highlighted candidate.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][int]$Index)

    if ($script:Candidates.Count -eq 0) { return }
    $candidate = $script:Candidates[$Index]

    [Microsoft.PowerShell.PSConsoleReadLine]::RevertLine()
    [Microsoft.PowerShell.PSConsoleReadLine]::Insert($candidate.command)

    # Destructive suggestions are marked because the command lands in the
    # buffer ready to run and Enter is one keystroke away.
    if ($candidate.PSObject.Properties.Name -contains 'dangerous' -and
        $candidate.dangerous) {
        $position = if ($script:Candidates.Count -gt 1) {
            " [$($Index + 1)/$($script:Candidates.Count)]"
        } else { '' }
        Write-Host ''
        Write-Host "  destructive$position" -ForegroundColor Red -NoNewline
        Write-Host ''
    }
}

function Invoke-ClAiComplete {
    <#
        .SYNOPSIS
        The Tab handler.

        .DESCRIPTION
        First Tab asks the daemon. Subsequent Tabs move through the candidates
        already fetched, so cycling costs nothing. If there is no suggestion,
        the built-in completion runs instead -- the user should not be able to
        tell that cl-ai considered the line and declined.
    #>
    param($key, $arg)

    $line   = $null
    $cursor = $null
    [Microsoft.PowerShell.PSConsoleReadLine]::GetBufferState([ref]$line, [ref]$cursor)

    # Already cycling: advance rather than re-query.
    if ($script:Candidates.Count -gt 0 -and $line -eq $script:Candidates[$script:Index].command) {
        $script:Index = ($script:Index + 1) % $script:Candidates.Count
        Show-ClAiCandidate -Index $script:Index
        return
    }

    Reset-ClAiCycle
    $suggestions = Get-ClAiSuggestions -Buffer $line

    if ($suggestions.Count -eq 0) {
        [Microsoft.PowerShell.PSConsoleReadLine]::TabCompleteNext()
        return
    }

    $script:Candidates   = $suggestions
    $script:OriginalLine = $line
    $script:Index        = 0
    Show-ClAiCandidate -Index 0
}

function Invoke-ClAiNext {
    param($key, $arg)
    if ($script:Candidates.Count -eq 0) {
        [Microsoft.PowerShell.PSConsoleReadLine]::NextHistory()
        return
    }
    $script:Index = ($script:Index + 1) % $script:Candidates.Count
    Show-ClAiCandidate -Index $script:Index
}

function Invoke-ClAiPrevious {
    param($key, $arg)
    if ($script:Candidates.Count -eq 0) {
        [Microsoft.PowerShell.PSConsoleReadLine]::PreviousHistory()
        return
    }
    $script:Index = ($script:Index - 1 + $script:Candidates.Count) % $script:Candidates.Count
    Show-ClAiCandidate -Index $script:Index
}

function Register-ClAiKeyHandlers {
    <#
        .SYNOPSIS
        Bind the keys. Safe to call twice.

        .DESCRIPTION
        Only binds when PSReadLine is actually loaded, so importing this module
        in a non-interactive session (a script, or CI) is harmless.
    #>
    [CmdletBinding()]
    param(
        [string]$CompleteKey = 'Tab',
        [string]$NextKey     = 'DownArrow',
        [string]$PreviousKey = 'UpArrow'
    )

    if (-not (Get-Module -ListAvailable -Name PSReadLine)) {
        Write-Verbose 'PSReadLine is not available; cl-ai key handlers not bound.'
        return $false
    }
    if (-not ([System.Management.Automation.PSTypeName]'Microsoft.PowerShell.PSConsoleReadLine').Type) {
        Write-Verbose 'PSReadLine is not loaded; cl-ai key handlers not bound.'
        return $false
    }

    Set-PSReadLineKeyHandler -Key $CompleteKey -ScriptBlock ${function:Invoke-ClAiComplete} `
        -BriefDescription 'clAiComplete' -Description 'Suggest a command with cl-ai'
    Set-PSReadLineKeyHandler -Key $NextKey -ScriptBlock ${function:Invoke-ClAiNext} `
        -BriefDescription 'clAiNext' -Description 'Next cl-ai suggestion'
    Set-PSReadLineKeyHandler -Key $PreviousKey -ScriptBlock ${function:Invoke-ClAiPrevious} `
        -BriefDescription 'clAiPrevious' -Description 'Previous cl-ai suggestion'
    return $true
}

Export-ModuleMember -Function @(
    'Register-ClAiKeyHandlers'
    'Invoke-ClAiComplete'
    'Invoke-ClAiNext'
    'Invoke-ClAiPrevious'
    'Get-ClAiSuggestions'
    'Invoke-ClAiRequest'
    'Get-ClAiPortFile'
    'Reset-ClAiCycle'
    'Show-ClAiCandidate'
)
