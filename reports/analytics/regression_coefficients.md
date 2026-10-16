| crop | model | term | unit | coef | se_cluster | t | p | ci_lo | ci_hi | pct_of_mean_yield_per_1sd_within | vif_within | n | n_districts | r2 | partial_r2_weather | synthetic_regressor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rice | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0242 | 0.0073 | 3.3150 | 0.0009 | 0.0099 | 0.0386 | 1.9645 | 1.1369 | 918 | 51 | 0.8085 | 0.0057 | False |
| rice | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.1068 | 0.0589 | -1.8114 | 0.0701 | -0.2223 | 0.0088 | -1.8738 | 1.1369 | 918 | 51 | 0.8085 | 0.0057 | False |
| rice | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | 0.0200 | 0.0085 | 2.3629 | 0.0181 | 0.0034 | 0.0366 | 1.6193 | 1.4680 | 918 | 51 | 0.8098 | 0.0069 | False |
| rice | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0729 | 0.0661 | -1.1030 | 0.2700 | -0.2024 | 0.0566 | -1.2795 | 1.2271 | 918 | 51 | 0.8098 | 0.0069 | False |
| rice | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | -0.0405 | 0.0372 | -1.0880 | 0.2766 | -0.1133 | 0.0324 | -0.8832 | 1.7398 | 918 | 51 | 0.8098 | 0.0069 | True |
| rice | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.1181 | 0.0638 | 1.8517 | 0.0641 | -0.0069 | 0.2431 | 1.7968 | 2.0051 | 918 | 51 | 0.8098 | 0.0069 | True |
| rice | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0217 | 0.0074 | 2.9426 | 0.0033 | 0.0072 | 0.0361 | 1.7552 | 1.2050 | 918 | 51 | 0.7872 | 0.0143 | False |
| rice | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.1473 | 0.0248 | -5.9445 | 0.0000 | -0.1959 | -0.0988 | -2.5860 | 1.2050 | 918 | 51 | 0.7872 | 0.0143 | False |
| wheat | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | -0.0368 | 0.1022 | -0.3599 | 0.7189 | -0.2372 | 0.1636 | -1.6442 | 1.1856 | 315 | 30 | 0.4940 | 0.0060 | False |
| wheat | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | 0.1446 | 0.1573 | 0.9197 | 0.3577 | -0.1636 | 0.4528 | 7.4821 | 1.1856 | 315 | 30 | 0.4940 | 0.0060 | False |
| wheat | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0106 | 0.1169 | -0.0903 | 0.9280 | -0.2396 | 0.2185 | -0.4717 | 1.9948 | 315 | 30 | 0.4972 | 0.0092 | False |
| wheat | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | 0.1710 | 0.1585 | 1.0789 | 0.2806 | -0.1397 | 0.4818 | 8.8489 | 1.3429 | 315 | 30 | 0.4972 | 0.0092 | False |
| wheat | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.0537 | 0.0390 | 1.3767 | 0.1686 | -0.0227 | 0.1301 | 5.2336 | 1.7417 | 315 | 30 | 0.4972 | 0.0092 | True |
| wheat | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | -0.1477 | 0.1677 | -0.8812 | 0.3782 | -0.4763 | 0.1809 | -3.0587 | 3.0060 | 315 | 30 | 0.4972 | 0.0092 | True |
| wheat | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0376 | 0.0471 | 0.7985 | 0.4246 | -0.0547 | 0.1298 | 1.6787 | 1.2381 | 315 | 30 | 0.4189 | 0.0020 | False |
| wheat | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | 0.0409 | 0.0839 | 0.4876 | 0.6259 | -0.1235 | 0.2054 | 2.1165 | 1.2381 | 315 | 30 | 0.4189 | 0.0020 | False |
| cotton | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0001 | 0.0031 | 0.0270 | 0.9785 | -0.0060 | 0.0061 | 0.0541 | 1.1512 | 772 | 45 | 0.5815 | 0.0104 | False |
| cotton | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.0580 | 0.0173 | -3.3646 | 0.0008 | -0.0919 | -0.0242 | -8.6920 | 1.1512 | 772 | 45 | 0.5815 | 0.0104 | False |
| cotton | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0027 | 0.0037 | -0.7350 | 0.4623 | -0.0100 | 0.0045 | -1.7676 | 1.5661 | 772 | 45 | 0.5861 | 0.0150 | False |
| cotton | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.0465 | 0.0179 | -2.6062 | 0.0092 | -0.0815 | -0.0115 | -6.9678 | 1.2416 | 772 | 45 | 0.5861 | 0.0150 | False |
| cotton | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | -0.0035 | 0.0114 | -0.3062 | 0.7595 | -0.0258 | 0.0188 | -0.6468 | 1.7358 | 772 | 45 | 0.5861 | 0.0150 | True |
| cotton | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.0382 | 0.0176 | 2.1696 | 0.0300 | 0.0037 | 0.0727 | 4.7701 | 2.0633 | 772 | 45 | 0.5861 | 0.0150 | True |
| cotton | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | 0.0006 | 0.0027 | 0.2390 | 0.8111 | -0.0046 | 0.0059 | 0.4178 | 1.2174 | 772 | 45 | 0.5322 | 0.0215 | False |
| cotton | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.0433 | 0.0085 | -5.0819 | 0.0000 | -0.0600 | -0.0266 | -6.4873 | 1.2174 | 772 | 45 | 0.5322 | 0.0215 | False |
| sugarcane | M1_twoway_FE_weather | rain_100mm | per +100 mm season rainfall | 0.0052 | 0.0330 | 0.1581 | 0.8743 | -0.0595 | 0.0699 | 0.1529 | 1.1148 | 839 | 50 | 0.5868 | 0.0021 | False |
| sugarcane | M1_twoway_FE_weather | tmax_c | per +1 C mean Tmax | -0.4547 | 0.2435 | -1.8672 | 0.0619 | -0.9319 | 0.0226 | -2.9285 | 1.1148 | 839 | 50 | 0.5868 | 0.0021 | False |
| sugarcane | M2_twoway_FE_weather_satellite | rain_100mm | per +100 mm season rainfall | -0.0371 | 0.0437 | -0.8477 | 0.3966 | -0.1228 | 0.0486 | -1.0860 | 1.4648 | 839 | 50 | 0.5903 | 0.0056 | False |
| sugarcane | M2_twoway_FE_weather_satellite | tmax_c | per +1 C mean Tmax | -0.2884 | 0.2453 | -1.1756 | 0.2398 | -0.7693 | 0.1924 | -1.8578 | 1.2042 | 839 | 50 | 0.5903 | 0.0056 | False |
| sugarcane | M2_twoway_FE_weather_satellite | ndvi_01 | per +0.1 peak NDVI | 0.0177 | 0.1383 | 0.1283 | 0.8979 | -0.2534 | 0.2888 | 0.1456 | 1.7132 | 839 | 50 | 0.5903 | 0.0056 | True |
| sugarcane | M2_twoway_FE_weather_satellite | soilm_01 | per +0.1 m3/m3 soil moisture | 0.5013 | 0.2893 | 1.7331 | 0.0831 | -0.0656 | 1.0683 | 2.8263 | 1.9884 | 839 | 50 | 0.5903 | 0.0056 | True |
| sugarcane | M3_district_FE_trend_weather | rain_100mm | per +100 mm season rainfall | -0.0204 | 0.0336 | -0.6066 | 0.5441 | -0.0862 | 0.0454 | -0.5965 | 1.1932 | 839 | 50 | 0.4805 | 0.0018 | False |
| sugarcane | M3_district_FE_trend_weather | tmax_c | per +1 C mean Tmax | -0.2544 | 0.1234 | -2.0616 | 0.0392 | -0.4962 | -0.0125 | -1.6385 | 1.1932 | 839 | 50 | 0.4805 | 0.0018 | False |
