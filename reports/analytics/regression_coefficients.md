| crop | model | term | unit | coef | se_cluster | t | p | ci_lo | ci_hi | pct_of_mean_yield_per_1sd_within | vif_within | n | n_districts | r2 | partial_r2_weather | synthetic_regressor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rice | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0056 | 0.0058 | 0.9633 | 0.3354 | -0.0058 | 0.0170 | 0.4654 | 1.3034 | 918 | 51 | 0.8070 | 0.0041 | False |
| rice | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.0874 | 0.0317 | -2.7568 | 0.0058 | -0.1496 | -0.0253 | -2.4519 | 1.3034 | 918 | 51 | 0.8070 | 0.0041 | False |
| rice | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | 0.0083 | 0.0060 | 1.3996 | 0.1616 | -0.0033 | 0.0200 | 0.6933 | 1.3329 | 918 | 51 | 0.8086 | 0.0058 | False |
| rice | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.1116 | 0.0513 | -2.1778 | 0.0294 | -0.2121 | -0.0112 | -3.1312 | 4.0826 | 918 | 51 | 0.8086 | 0.0058 | False |
| rice | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | -0.1005 | 0.0312 | -3.2173 | 0.0013 | -0.1617 | -0.0393 | -1.6822 | 1.6262 | 918 | 51 | 0.8086 | 0.0058 | True |
| rice | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.0342 | 0.0840 | 0.4078 | 0.6834 | -0.1303 | 0.1988 | 0.4714 | 3.7845 | 918 | 51 | 0.8086 | 0.0058 | True |
| rice | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | -0.0009 | 0.0061 | -0.1532 | 0.8783 | -0.0128 | 0.0110 | -0.0773 | 1.5252 | 918 | 51 | 0.7841 | 0.0112 | False |
| rice | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.1161 | 0.0219 | -5.3071 | 0.0000 | -0.1590 | -0.0732 | -3.2573 | 1.5252 | 918 | 51 | 0.7841 | 0.0112 | False |
| wheat | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | -0.0869 | 0.1024 | -0.8484 | 0.3962 | -0.2877 | 0.1139 | -3.6852 | 1.1856 | 315 | 30 | 0.4899 | 0.0019 | False |
| wheat | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.0055 | 0.0532 | -0.1026 | 0.9183 | -0.1097 | 0.0988 | -0.5491 | 1.1856 | 315 | 30 | 0.4899 | 0.0019 | False |
| wheat | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0484 | 0.1170 | -0.4133 | 0.6794 | -0.2777 | 0.1810 | -2.0508 | 1.6893 | 315 | 30 | 0.4910 | 0.0030 | False |
| wheat | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0119 | 0.0640 | -0.1857 | 0.8527 | -0.1373 | 0.1135 | -1.1958 | 2.5226 | 315 | 30 | 0.4910 | 0.0030 | False |
| wheat | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.0337 | 0.0366 | 0.9206 | 0.3573 | -0.0381 | 0.1055 | 3.2100 | 1.8977 | 315 | 30 | 0.4910 | 0.0030 | True |
| wheat | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | -0.1918 | 0.1636 | -1.1723 | 0.2411 | -0.5124 | 0.1289 | -4.1940 | 3.3295 | 315 | 30 | 0.4910 | 0.0030 | True |
| wheat | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0175 | 0.0396 | 0.4428 | 0.6579 | -0.0600 | 0.0950 | 0.7427 | 1.4513 | 315 | 30 | 0.4174 | 0.0005 | False |
| wheat | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.0042 | 0.0367 | -0.1142 | 0.9091 | -0.0761 | 0.0678 | -0.4218 | 1.4513 | 315 | 30 | 0.4174 | 0.0005 | False |
| cotton | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | -0.0026 | 0.0025 | -1.0507 | 0.2934 | -0.0075 | 0.0023 | -1.5961 | 1.4516 | 772 | 45 | 0.5806 | 0.0095 | False |
| cotton | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.0304 | 0.0076 | -4.0256 | 0.0001 | -0.0452 | -0.0156 | -7.1034 | 1.4516 | 772 | 45 | 0.5806 | 0.0095 | False |
| cotton | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0019 | 0.0022 | -0.8715 | 0.3835 | -0.0063 | 0.0024 | -1.1726 | 1.4975 | 772 | 45 | 0.5887 | 0.0176 | False |
| cotton | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0604 | 0.0162 | -3.7314 | 0.0002 | -0.0922 | -0.0287 | -14.1096 | 4.0543 | 772 | 45 | 0.5887 | 0.0176 | False |
| cotton | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | -0.0296 | 0.0107 | -2.7625 | 0.0057 | -0.0506 | -0.0086 | -4.1957 | 1.6358 | 772 | 45 | 0.5887 | 0.0176 | True |
| cotton | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | -0.0353 | 0.0290 | -1.2175 | 0.2234 | -0.0920 | 0.0215 | -3.9545 | 3.5937 | 772 | 45 | 0.5887 | 0.0176 | True |
| cotton | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | -0.0025 | 0.0024 | -1.0637 | 0.2875 | -0.0072 | 0.0021 | -1.5294 | 1.7260 | 772 | 45 | 0.5335 | 0.0228 | False |
| cotton | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.0325 | 0.0054 | -6.0458 | 0.0000 | -0.0430 | -0.0219 | -7.5812 | 1.7260 | 772 | 45 | 0.5335 | 0.0228 | False |
| sugarcane | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | -0.0732 | 0.0369 | -1.9845 | 0.0472 | -0.1455 | -0.0009 | -2.0564 | 1.3401 | 839 | 50 | 0.5878 | 0.0031 | False |
| sugarcane | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.2834 | 0.1378 | -2.0567 | 0.0397 | -0.5535 | -0.0133 | -2.8938 | 1.3401 | 839 | 50 | 0.5878 | 0.0031 | False |
| sugarcane | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0609 | 0.0364 | -1.6715 | 0.0946 | -0.1323 | 0.0105 | -1.7111 | 1.3739 | 839 | 50 | 0.5905 | 0.0058 | False |
| sugarcane | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.3515 | 0.1918 | -1.8325 | 0.0669 | -0.7274 | 0.0244 | -3.5885 | 4.1513 | 839 | 50 | 0.5905 | 0.0058 | False |
| sugarcane | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | -0.3867 | 0.1625 | -2.3804 | 0.0173 | -0.7051 | -0.0683 | -2.4156 | 1.6092 | 839 | 50 | 0.5905 | 0.0058 | True |
| sugarcane | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.1843 | 0.3328 | 0.5537 | 0.5798 | -0.4680 | 0.8365 | 0.9256 | 3.7747 | 839 | 50 | 0.5905 | 0.0058 | True |
| sugarcane | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | -0.0708 | 0.0343 | -2.0655 | 0.0389 | -0.1380 | -0.0036 | -1.9893 | 1.5918 | 839 | 50 | 0.4818 | 0.0031 | False |
| sugarcane | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.2194 | 0.0825 | -2.6586 | 0.0078 | -0.3812 | -0.0577 | -2.2401 | 1.5918 | 839 | 50 | 0.4818 | 0.0031 | False |
