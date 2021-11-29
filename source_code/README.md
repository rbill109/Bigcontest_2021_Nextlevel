
---
---
# data
## 1. 대회 제공 데이터
- 음식물 쓰레기 데이터
- 카드 데이터
- 거주인구 데이터 (7, 8월 데이터 추가 수집 가능)
- 유동인구 데이터

<br>

## 2. 외부 데이터
- 기상 데이터 (출처: [기상자료개방포털](https://data.kma.go.kr/cmmn/main.do))
- 검색량 데이터 (출처: [Naver 데이터랩](https://datalab.naver.com/keyword/trendSearch.naver))
- 세대수 및 인구밀도 데이터 (출처: [통계청](https://kostat.go.kr/portal/korea/index.action))
- 연령별 거주인구 데이터 (출처: [공공데이터포털](https://www.data.go.kr/index.do))
- 입도객 데이터 (출처: [서울열린데이터광장](http://li-st.com/reference/253))
- 공휴일 데이터 (출처: [공공데이터포털](https://www.data.go.kr/index.do))
---
---
# source_code
## A-1. 제공데이터_병합
대회 제공 데이터 전처리 및 병합 <br>
### merged_day_final.csv <br>
-> (50522, 47) <br>
-> column 확인
#### <음식물 쓰레기>
- base_date, emd_nm, em_g, em_area_cd_cnt, city, cluster, key

#### <카드>
- 유형2_cnt, 유형2_amt, 유형2_ratio, 배달_cnt, 배달_amt, 배달_ratio, 농축수산물_cnt, 농축수산물_amt, 농축수산물_ratio, 유형1_cnt, 유형1_amt, 유형1_ratio, 유형3_cnt, 유형3_amt, 유형3_ratio

#### <거주인구>
- korean_resd_pop, foreign_resd_pop, total_resd_pop, foreign_resd_ratio

#### <단기체류 외국인 유동인구>
- visit_short, CHN_short, JPN_short, USA_short, CHN_ratio_short, NotCHN_short

#### <장기체류 외국인 유동인구>
- resd_jeju, country_1_all_jeju, country_2_resd_jeju, country_2_visit_jeju, country_3_resd_jeju, country_4_resd_jeju, country_7_all_jeju, country_3_all_etc

#### <내국인 유동인구>
- resd_pop_cnt, work_pop_jeju, work_pop_etc, visit_pop_jeju, visit_pop_etc, vpc_2050_daytm, vpc_daytm_rt
---
## A-2. 기상데이터_수집
기상, 강수, 기온, 습도, 풍속 데이터 수집 및 전처리

## A-3. 검색데이터_수집
네이버 데이터랩 검색량 데이터 수집 및 전처리

## A-4. 태풍데이터_수집
태풍 발생 데이터 수집 및 전처리 (병합 포함)

---

## A-5. 외부데이터병합(최종병합)
기상, 검색, 미세먼지, 세대, 연령별 거주, 입도객 데이터 기존 데이터에 병합 <br>
### **▶ final_merged_by_day_v7.csv** <br>
7, 8월이 누락된 카드, 유동인구 변수를 제외한 데이터

2018.01.01 ~ 2021.06.30 <br>
-> (49609, 81) <br>
-> column 확인(추가한 column만 명시)
#### <기상>
- 강수, 기온, 습도, 풍속, 태풍, 미세먼지
#### <검색>
- 검색_관광, 검색_쇼핑, 검색_쓰레기, 검색_한파, 검색_강풍, 검색_태풍, 검색_장마, 검색_폭염, 검색_우박, 검색_날씨, 검색_미세먼지, 검색_황사, 읍면동별검색_생활, 읍면동별검색_관광, 읍면동별검색_쇼핑
#### <세대>
- fam, pop_per_fam, popdens
#### <입도객>
- 총계_월, 내국인전체_월, 외국인_월
#### <연령별 거주>
- 연령0020, 연령2030, 연령2040_rt, 연령3040, 연령4060, 연령60이상
#### <공휴일>
- holidays

<br>

### **▶ final_merged_by_day_v7_7_8.csv** 
7, 8월 수집이 가능한 거주 변수 및 외부 데이터<br>
2021.07.01 ~ 2021.08.31  <br>
-> (2542, 40) <br>

---
## B-1. 7_8월_X_예측

## C-1. Prophet_7, 8월 예측값(최종)

## C-2.  7_8월_Y_예측(추가)

## C-3.  7_8월_Y_예측(알수없음)

