# Delayed-start simulation for the b=8 cell (need >= 400 pure iterations captured, 0 straddling)

## e1_dry_off_v2: replay span 51.193 s; b=8 wave first iteration 181 at 0.037 s, first pure 198 at 3.608 s, last pure end 8.818 s, pure n=1008

| delay s | window | gate | pure captured | straddling | captured total | first..last captured |
|---|---|---|---|---|---|---|
| none | [-inf, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 0.0 | [0.0, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 1.0 | [1.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2155 | 186..2342 |
| 2.0 | [2.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2151 | 190..2342 |
| 3.0 | [3.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2146 | 195..2342 |
| 3.5 | [3.5, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2144 | 197..2342 |
| 3.6 | [3.6, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2143 | 198..2342 |
| 3.7 | [3.7, inf] | FAIL | 990 | 1 {'pure_b8': 1} | 2125 | 216..2342 |
| 4.0 | [4.0, inf] | FAIL | 932 | 1 {'pure_b8': 1} | 2067 | 274..2342 |
| 10.0 | [10.0, inf] | FAIL | 0 | 0  | 1120 | 1223..2342 |

## e1_dry_on_v2: replay span 50.692 s; b=8 wave first iteration 181 at 0.035 s, first pure 198 at 3.617 s, last pure end 8.981 s, pure n=1008

| delay s | window | gate | pure captured | straddling | captured total | first..last captured |
|---|---|---|---|---|---|---|
| none | [-inf, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 0.0 | [0.0, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 1.0 | [1.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2155 | 186..2342 |
| 2.0 | [2.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2151 | 190..2342 |
| 3.0 | [3.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2146 | 195..2342 |
| 3.5 | [3.5, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2144 | 197..2342 |
| 3.6 | [3.6, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2143 | 198..2342 |
| 3.7 | [3.7, inf] | FAIL | 993 | 0  | 2128 | 213..2342 |
| 4.0 | [4.0, inf] | FAIL | 936 | 1 {'pure_b8': 1} | 2071 | 270..2342 |
| 10.0 | [10.0, inf] | FAIL | 0 | 0  | 1120 | 1223..2342 |

## e1_dry_off2_v2: replay span 51.189 s; b=8 wave first iteration 181 at 0.033 s, first pure 198 at 3.612 s, last pure end 8.835 s, pure n=1008

| delay s | window | gate | pure captured | straddling | captured total | first..last captured |
|---|---|---|---|---|---|---|
| none | [-inf, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 0.0 | [0.0, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 1.0 | [1.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2155 | 186..2342 |
| 2.0 | [2.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2151 | 190..2342 |
| 3.0 | [3.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2146 | 195..2342 |
| 3.5 | [3.5, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2144 | 197..2342 |
| 3.6 | [3.6, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2143 | 198..2342 |
| 3.7 | [3.7, inf] | FAIL | 991 | 1 {'pure_b8': 1} | 2126 | 215..2342 |
| 4.0 | [4.0, inf] | FAIL | 933 | 1 {'pure_b8': 1} | 2068 | 273..2342 |
| 10.0 | [10.0, inf] | FAIL | 0 | 0  | 1120 | 1223..2342 |

## e1_nsys_val: replay span 51.197 s; b=8 wave first iteration 181 at 0.033 s, first pure 198 at 3.723 s, last pure end 8.984 s, pure n=1008

| delay s | window | gate | pure captured | straddling | captured total | first..last captured |
|---|---|---|---|---|---|---|
| none | [-inf, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 0.0 | [0.0, inf] | PASS | 1008 | 0  | 2160 | 181..2342 |
| 1.0 | [1.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2155 | 186..2342 |
| 2.0 | [2.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2151 | 190..2342 |
| 3.0 | [3.0, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2146 | 195..2342 |
| 3.5 | [3.5, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2144 | 197..2342 |
| 3.6 | [3.6, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2143 | 198..2342 |
| 3.7 | [3.7, inf] | FAIL | 1008 | 1 {'mixed_b8': 1} | 2143 | 198..2342 |
| 4.0 | [4.0, inf] | FAIL | 955 | 1 {'pure_b8': 1} | 2090 | 251..2342 |
| 10.0 | [10.0, inf] | FAIL | 0 | 0  | 1120 | 1223..2342 |

