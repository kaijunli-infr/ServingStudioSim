| quantity | none (V3 as measured) | intermediate ~49 M | full E1 ~216 M | full E1, rows 173 M | full E1, rows 259 M | V3 limit (unchanged) |
|---|---|---|---|---|---|---|
| kernel rows | 24.5 M (1.00 × V3) | 49.0 M (2.00 × V3) | 216.0 M (8.82 × V3) | 173.0 M (7.06 × V3) | 259.0 M (10.57 × V3) | — |
| flush, s | 112 – 112 (stress 112) | 203 – 224 (stress 249) | 721 – 987 (stress 1369) | 596 – 791 (stress 1060) | 842 – 1184 (stress 1686) | 300 |
| import, s | 53 – 53 (stress 53) | 97 – 106 (stress 118) | 356 – 467 (stress 648) | 293 – 374 (stress 502) | 417 – 560 (stress 798) | 300 |
| stop → report, s | 166 – 166 (stress 166) | 301 – 331 (stress 367) | 1078 – 1456 (stress 2017) | 890 – 1166 (stress 1563) | 1260 – 1745 (stress 2485) | 600 |
| GPU-held, s | 1797 – 1950 (stress 1950) | 2092 – 2275 (stress 2311) | 3466 – 3997 (stress 4558) | 3278 – 3707 (stress 4104) | 3648 – 4286 (stress 5026) | 2700 |
| GPU-held if the 600 s hard deadline acts, s | 2439 | 2599 | 3196 | 3196 | 3196 | 2700 |
| reservation as in V3 (held through closeout), min | 60 – 62 (stress 62) | 75 – 78 (stress 81) | 166 – 176 (stress 220) | 145 – 154 (stress 185) | 187 – 199 (stress 257) | 90 |
| reservation if deleted at verified GPU release, min | 37 – 40 (stress 40) | 42 – 45 (stress 46) | 65 – 74 (stress 83) | 62 – 69 (stress 76) | 68 – 79 (stress 91) | 90 |
| importer peak memory, GiB | 15.6 – 15.6 (stress 15.6) | 28.0 – 31.2 (stress 34.6) | 97.9 – 137.6 (stress 190.7) | 81.2 – 110.2 (stress 147.8) | 114.1 – 165.0 (stress 235.0) | 128 |
| MemAvailable minimum, GiB | 547 – 547 (stress 547) | 532 – 535 (stress 528) | 425 – 465 (stress 372) | 453 – 482 (stress 415) | 398 – 449 (stress 328) | floor 64 |
| raw stream, GB | 0.66 – 0.66 (stress 0.66) | 1.26 – 1.32 (stress 1.47) | 4.96 – 5.83 (stress 8.08) | 4.04 – 4.67 (stress 6.26) | 5.87 – 6.99 (stress 9.96) | — |
| report, GB | 0.79 – 0.79 (stress 0.79) | 1.58 – 1.60 (stress 1.76) | 6.98 – 7.22 (stress 9.68) | 5.59 – 5.76 (stress 7.50) | 8.37 – 8.68 (stress 11.92) | — |
| in-session export, GB | 4.0 – 4.0 (stress 4.0) | 7.9 – 8.0 (stress 8.8) | 34.9 – 35.6 (stress 48.3) | 27.9 – 28.5 (stress 37.4) | 41.8 – 42.8 (stress 59.5) | — |
| temporary peak (post-run, with retained streams), GB | 5.9 – 5.9 (stress 5.9) | 10.4 – 10.5 (stress 11.4) | 41.2 – 42.0 (stress 56.6) | 33.3 – 33.9 (stress 44.1) | 49.1 – 50.2 (stress 69.4) | 10 |
| run root, GB | 6.3 – 6.3 (stress 6.3) | 12.6 – 12.6 (stress 14.0) | 55.4 – 55.6 (stress 77.1) | 44.4 – 44.6 (stress 59.7) | 66.4 – 66.7 (stress 95.0) | 8 |
| run root + archive, GB | 13.3 – 13.3 (stress 13.3) | 26.5 – 26.6 (stress 29.5) | 116.6 – 117.1 (stress 162.4) | 93.4 – 93.8 (stress 125.8) | 139.8 – 140.5 (stress 200.0) | 16 |
| export and parse after release, s | 613 – 613 (stress 613) | 1220 – 1226 (stress 1360) | 5322 – 5404 (stress 7490) | 4269 – 4328 (stress 5803) | 6373 – 6480 (stress 9230) | — |
| offline import, s | 56 – 56 (stress 56) | 112 – 121 (stress 124) | 494 – 623 (stress 684) | 395 – 487 (stress 530) | 592 – 762 (stress 843) | — |
| offline export, s | 130 – 130 (stress 130) | 260 – 262 (stress 288) | 1146 – 1170 (stress 1589) | 918 – 935 (stress 1231) | 1374 – 1406 (stress 1957) | — |

Largest size at which each hypothesis stays inside the unchanged V3 limit (M rows):

| limit | power | line | prop | stress |
|---|---|---|---|---|
| flush_s ≤ 300 | 78 | 70 | 66 | 58 |
| import_s ≤ 300 | 178 | 150 | 139 | 111 |
| importer_rss_gib ≤ 128 | 297 | 224 | 201 | 153 |
| run_root_gb ≤ 8 | 31 | 31 | 31 | 30 |

Exponent that the V2 and V3 points happen to give (two points; not a validated exponent): flush_s 0.86, import_s 0.87, importer_rss_gib 0.84, stream_gb 0.93, report_gb 1.02, export_gb 1.01, run_root_gb 1.00, post_release_export_parse_s 0.99, offline_import_s 1.11

Check against the one uncontrolled observation (e1_main_v3 flush 865 s at an estimated 216 M rows): power 721 s, line 895 s, prop 987 s, stress 1369 s. Exponent implied by V3 and that observation: 0.94 at 216 M, 1.05 at 173 M, 0.87 at 259 M.
