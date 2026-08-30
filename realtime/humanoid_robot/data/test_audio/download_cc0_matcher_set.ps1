$ErrorActionPreference = 'Stop'

$outputDir = Join-Path $PSScriptRoot 'cc0_matcher_set'
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$tracks = @(
    @{ File = '01_electronic_backbeat.mp3'; Source = 'Electronic/Backbeat.mp3' },
    @{ File = '02_hiphop_hippety_hop.mp3'; Source = 'Electronic/Hippety Hop.mp3' },
    @{ File = '03_jazz_bebop_for_joey.mp3'; Source = 'Comedy/BeBop for Joey.mp3' },
    @{ File = '04_world_bollywood_groove.mp3'; Source = 'World/Bollywood Groove.mp3' },
    @{ File = '05_latin_cumbish.mp3'; Source = 'World/Cumbish.mp3' },
    @{ File = '06_upbeat_bar_brawl.mp3'; Source = 'Upbeat/Bar Brawl.mp3' },
    @{ File = '07_orchestral_battle_ready.mp3'; Source = 'Epic/Battle Ready.mp3' },
    @{ File = '08_waltz_isolation_waltz.mp3'; Source = 'Romance/Isolation Waltz.mp3' },
    @{ File = '09_funk_funky_energy_loop.mp3'; Source = 'Scoring/Funky Energy Loop.mp3' },
    @{ File = '10_ambient_alien_spaceship.mp3'; Source = 'Horror/Alien Spaceship Atmosphere.mp3' }
)

foreach ($track in $tracks) {
    $hex = [Convert]::ToHexString([Text.Encoding]::UTF8.GetBytes($track.Source)).ToLowerInvariant()
    $uri = "https://en.freepd.cn/api/music/$hex"
    $destination = Join-Path $outputDir $track.File

    Write-Host "Downloading $($track.File)"
    Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $destination
}

Write-Host "Downloaded $($tracks.Count) CC0 tracks to $outputDir"
