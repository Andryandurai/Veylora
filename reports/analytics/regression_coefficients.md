| crop | model | term | unit | coef | se_cluster | t | p | ci_lo | ci_hi | pct_of_mean_yield_per_1sd_within | vif_within | n | n_districts | r2 | partial_r2_weather | synthetic_regressor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rice | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0242 | 0.0073 | 3.3150 | 0.0009 | 0.0099 | 0.0386 | 1.9645 | 1.1369 | 918 | 51 | 0.8085 | 0.0057 | False |
| rice | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.1068 | 0.0589 | -1.8114 | 0.0701 | -0.2223 | 0.0088 | -1.8738 | 1.1369 | 918 | 51 | 0.8085 | 0.0057 | False |
| rice | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | 0.0238 | 0.0072 | 3.3203 | 0.0009 | 0.0098 | 0.0379 | 1.9305 | 1.1447 | 918 | 51 | 0.8088 | 0.0059 | False |
| rice | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0780 | 0.0704 | -1.1082 | 0.2678 | -0.2161 | 0.0600 | -1.3697 | 2.0884 | 918 | 51 | 0.8088 | 0.0059 | False |
| rice | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.0369 | 0.0632 | 0.5835 | 0.5595 | -0.0870 | 0.1608 | 0.4667 | 1.1836 | 918 | 51 | 0.8088 | 0.0059 | False |
| rice | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.0488 | 0.1549 | 0.3153 | 0.7526 | -0.2548 | 0.3525 | 0.4049 | 2.1405 | 918 | 51 | 0.8088 | 0.0059 | False |
| rice | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0217 | 0.0074 | 2.9426 | 0.0033 | 0.0072 | 0.0361 | 1.7552 | 1.2050 | 918 | 51 | 0.7872 | 0.0143 | False |
| rice | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.1473 | 0.0248 | -5.9445 | 0.0000 | -0.1959 | -0.0988 | -2.5860 | 1.2050 | 918 | 51 | 0.7872 | 0.0143 | False |
| wheat | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | -0.0368 | 0.1022 | -0.3599 | 0.7189 | -0.2372 | 0.1636 | -1.6442 | 1.1856 | 315 | 30 | 0.4940 | 0.0060 | False |
| wheat | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | 0.1446 | 0.1573 | 0.9197 | 0.3577 | -0.1636 | 0.4528 | 7.4821 | 1.1856 | 315 | 30 | 0.4940 | 0.0060 | False |
| wheat | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | 0.0139 | 0.0941 | 0.1475 | 0.8827 | -0.1706 | 0.1984 | 0.6205 | 1.4155 | 315 | 30 | 0.5068 | 0.0188 | False |
| wheat | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | 0.2394 | 0.1360 | 1.7595 | 0.0785 | -0.0273 | 0.5060 | 12.3837 | 1.7838 | 315 | 30 | 0.5068 | 0.0188 | False |
| wheat | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.2023 | 0.1122 | 1.8028 | 0.0714 | -0.0176 | 0.4221 | 9.2723 | 1.4461 | 315 | 30 | 0.5068 | 0.0188 | False |
| wheat | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | -0.3392 | 0.2114 | -1.6047 | 0.1086 | -0.7536 | 0.0751 | -6.2768 | 1.7267 | 315 | 30 | 0.5068 | 0.0188 | False |
| wheat | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0376 | 0.0471 | 0.7985 | 0.4246 | -0.0547 | 0.1298 | 1.6787 | 1.2381 | 315 | 30 | 0.4189 | 0.0020 | False |
| wheat | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | 0.0409 | 0.0839 | 0.4876 | 0.6259 | -0.1235 | 0.2054 | 2.1165 | 1.2381 | 315 | 30 | 0.4189 | 0.0020 | False |
| cotton | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0001 | 0.0031 | 0.0270 | 0.9785 | -0.0060 | 0.0061 | 0.0541 | 1.1512 | 772 | 45 | 0.5815 | 0.0104 | False |
| cotton | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.0580 | 0.0173 | -3.3646 | 0.0008 | -0.0919 | -0.0242 | -8.6920 | 1.1512 | 772 | 45 | 0.5815 | 0.0104 | False |
| cotton | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0005 | 0.0029 | -0.1747 | 0.8613 | -0.0063 | 0.0053 | -0.3347 | 1.1642 | 772 | 45 | 0.5907 | 0.0196 | False |
| cotton | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0049 | 0.0214 | -0.2305 | 0.8177 | -0.0469 | 0.0370 | -0.7383 | 2.1289 | 772 | 45 | 0.5907 | 0.0196 | False |
| cotton | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.0089 | 0.0155 | 0.5741 | 0.5659 | -0.0215 | 0.0393 | 0.9244 | 1.1593 | 772 | 45 | 0.5907 | 0.0196 | False |
| cotton | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.1333 | 0.0368 | 3.6239 | 0.0003 | 0.0612 | 0.2054 | 9.3393 | 2.1822 | 772 | 45 | 0.5907 | 0.0196 | False |
| cotton | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0006 | 0.0027 | 0.2390 | 0.8111 | -0.0046 | 0.0059 | 0.4178 | 1.2174 | 772 | 45 | 0.5322 | 0.0215 | False |
| cotton | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.0433 | 0.0085 | -5.0819 | 0.0000 | -0.0600 | -0.0266 | -6.4873 | 1.2174 | 772 | 45 | 0.5322 | 0.0215 | False |
| sugarcane | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0052 | 0.0330 | 0.1581 | 0.8743 | -0.0595 | 0.0699 | 0.1529 | 1.1148 | 839 | 50 | 0.5868 | 0.0021 | False |
| sugarcane | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.4547 | 0.2435 | -1.8672 | 0.0619 | -0.9319 | 0.0226 | -2.9285 | 1.1148 | 839 | 50 | 0.5868 | 0.0021 | False |
| sugarcane | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | 0.0019 | 0.0333 | 0.0560 | 0.9553 | -0.0633 | 0.0670 | 0.0546 | 1.1262 | 839 | 50 | 0.5896 | 0.0049 | False |
| sugarcane | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | 0.0263 | 0.3661 | 0.0718 | 0.9428 | -0.6912 | 0.7438 | 0.1693 | 2.0874 | 839 | 50 | 0.5896 | 0.0049 | False |
| sugarcane | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.2047 | 0.2066 | 0.9906 | 0.3219 | -0.2003 | 0.6096 | 0.9517 | 1.1773 | 839 | 50 | 0.5896 | 0.0049 | False |
| sugarcane | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 1.0639 | 0.5282 | 2.0141 | 0.0440 | 0.0286 | 2.0992 | 3.2887 | 2.1464 | 839 | 50 | 0.5896 | 0.0049 | False |
| sugarcane | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | -0.0204 | 0.0336 | -0.6066 | 0.5441 | -0.0862 | 0.0454 | -0.5965 | 1.1932 | 839 | 50 | 0.4805 | 0.0018 | False |
| sugarcane | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.2544 | 0.1234 | -2.0616 | 0.0392 | -0.4962 | -0.0125 | -1.6385 | 1.1932 | 839 | 50 | 0.4805 | 0.0018 | False |
