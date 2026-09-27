# Dataset Audit Report — Amazon ML Challenge 2026: Business Entity Resolution

**Date:** 2026-09-26

**Status:** Comprehensive Data Audit Complete

## 1. File Statistics & Dimensions

| Dataset File | Rows | Columns | File Size (MB) |
| :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 4 | 200.34 MB |
| `train_source2.tsv` | 5,034,616 | 4 | 466.63 MB |
| `train_source3.tsv` | 5,285,603 | 4 | 480.37 MB |
| `train_ground_truth.tsv` | 2,206,821 | 2 | 121.13 MB |
| `test_source1.tsv` | 1,732,544 | 4 | 166.91 MB |
| `test_source2.tsv` | 4,887,273 | 4 | 485.86 MB |
| `test_source3.tsv` | 5,082,316 | 4 | 482.56 MB |


## 2. Column Names, Data Types, and Missing Values

### `train_source1`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 2,206,821 | 0 | 0.00% | 2,206,821 |
| `business_name` | str | 2,206,821 | 0 | 0.00% | 1,539,229 |
| `business_address` | str | 2,206,821 | 0 | 0.00% | 2,130,606 |
| `country` | str | 2,206,821 | 0 | 0.00% | 2 |


### `train_source2`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 5,034,616 | 0 | 0.00% | N/A (>3M) |
| `business_name` | str | 5,034,614 | 2 | 0.00% | N/A (>3M) |
| `business_address` | str | 4,865,649 | 168,967 | 3.36% | N/A (>3M) |
| `country` | str | 5,034,616 | 0 | 0.00% | N/A (>3M) |


### `train_source3`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 5,285,603 | 0 | 0.00% | N/A (>3M) |
| `business_name` | str | 5,285,590 | 13 | 0.00% | N/A (>3M) |
| `business_address` | str | 5,109,687 | 175,916 | 3.33% | N/A (>3M) |
| `country` | str | 5,285,603 | 0 | 0.00% | N/A (>3M) |


### `train_ground_truth`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `source1_entity_id` | str | 2,206,821 | 0 | 0.00% | 2,206,821 |
| `matched_entity_ids` | str | 2,083,574 | 123,247 | 5.58% | 2,083,574 |


### `test_source1`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 1,732,544 | 0 | 0.00% | 1,732,544 |
| `business_name` | str | 1,732,544 | 0 | 0.00% | 1,238,867 |
| `business_address` | str | 1,732,544 | 0 | 0.00% | 1,677,483 |
| `country` | str | 1,732,544 | 0 | 0.00% | 3 |


### `test_source2`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 4,887,273 | 0 | 0.00% | N/A (>3M) |
| `business_name` | str | 4,887,227 | 46 | 0.00% | N/A (>3M) |
| `business_address` | str | 4,757,865 | 129,408 | 2.65% | N/A (>3M) |
| `country` | str | 4,887,273 | 0 | 0.00% | N/A (>3M) |


### `test_source3`

| Column | Dtype | Non-Null Count | Null Count | % Missing | Unique Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `entity_id` | str | 5,082,316 | 0 | 0.00% | N/A (>3M) |
| `business_name` | str | 5,082,257 | 59 | 0.00% | N/A (>3M) |
| `business_address` | str | 4,946,218 | 136,098 | 2.68% | N/A (>3M) |
| `country` | str | 5,082,316 | 0 | 0.00% | N/A (>3M) |


## 3. Country Distribution Across Sources

### Country Breakdown in `train_source1` (Total: 2,206,821)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `US` | 1,323,633 | 59.98% |
| `India` | 883,188 | 40.02% |


### Country Breakdown in `train_source2` (Total: 5,034,616)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `US` | 3,016,817 | 59.92% |
| `India` | 2,017,799 | 40.08% |


### Country Breakdown in `train_source3` (Total: 5,285,603)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `US` | 3,170,056 | 59.98% |
| `India` | 2,115,547 | 40.02% |


### Country Breakdown in `test_source1` (Total: 1,732,544)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `India` | 809,986 | 46.75% |
| `US` | 663,106 | 38.27% |
| `France` | 259,452 | 14.98% |


### Country Breakdown in `test_source2` (Total: 4,887,273)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `India` | 2,312,565 | 47.32% |
| `US` | 1,871,330 | 38.29% |
| `France` | 703,378 | 14.39% |


### Country Breakdown in `test_source3` (Total: 5,082,316)

| Country | Count | Percentage |
| :--- | :--- | :--- |
| `India` | 2,405,000 | 47.32% |
| `US` | 1,945,701 | 38.28% |
| `France` | 731,615 | 14.40% |


## 4. Ground Truth Match Statistics

- **Total Source 1 Entities in Training Ground Truth:** 2,206,821
- **Total Match Links (S1 ↔ S2/S3 pairs):** 7,638,365
- **Total S2 Matches:** 3,693,619 (48.36% of links)
- **Total S3 Matches:** 3,944,746 (51.64% of links)
- **Singleton (Zero Match) S1 Entities:** 123,247 (5.58%)
- **Matched S1 Entities (>=1 match):** 2,083,574 (94.42%)

### Distribution of Number of Matches per Source 1 Entity

| Match Count | S1 Count | % of All S1 | Cumulative % |
| :--- | :--- | :--- | :--- |
| 0 matches | 123,247 | 5.58% | 5.58% |
| 1 matches | 119,157 | 5.40% | 10.98% |
| 2 matches | 375,212 | 17.00% | 27.99% |
| 3 matches | 530,841 | 24.05% | 52.04% |
| 4 matches | 484,115 | 21.94% | 73.98% |
| 5 matches | 321,957 | 14.59% | 88.57% |
| 6 matches | 164,868 | 7.47% | 96.04% |
| 7 matches | 63,968 | 2.90% | 98.94% |
| 8 matches | 18,680 | 0.85% | 99.78% |
| 9 matches | 4,205 | 0.19% | 99.97% |
| 10 matches | 534 | 0.02% | 100.00% |
| 11 matches | 37 | 0.00% | 100.00% |


## 5. Text Field Statistics (Name & Address Lengths)

### `train_source1`

- **`business_name` Character Length:** Min=3, Median=24.0, Mean=24.0, Max=71
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.5, Max=12
- **`business_address` Character Length:** Min=13, Median=41.0, Mean=52.1, Max=222
- **`business_address` Word Count:** Min=3, Median=7.0, Mean=8.0, Max=38


### `train_source2`

- **`business_name` Character Length:** Min=2, Median=25.0, Mean=25.1, Max=104
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.5, Max=15
- **`business_address` Character Length:** Min=11, Median=37.0, Mean=47.9, Max=204
- **`business_address` Word Count:** Min=2, Median=6.0, Mean=7.5, Max=33


### `train_source3`

- **`business_name` Character Length:** Min=2, Median=25.0, Mean=25.2, Max=80
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.5, Max=13
- **`business_address` Character Length:** Min=6, Median=42.0, Mean=48.4, Max=198
- **`business_address` Word Count:** Min=2, Median=6.0, Mean=7.4, Max=31


### `test_source1`

- **`business_name` Character Length:** Min=3, Median=24.0, Mean=23.8, Max=92
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.5, Max=13
- **`business_address` Character Length:** Min=15, Median=50.0, Mean=57.1, Max=216
- **`business_address` Word Count:** Min=2, Median=8.0, Mean=8.6, Max=34


### `test_source2`

- **`business_name` Character Length:** Min=2, Median=25.0, Mean=25.7, Max=74
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.6, Max=11
- **`business_address` Character Length:** Min=5, Median=43.0, Mean=51.7, Max=226
- **`business_address` Word Count:** Min=1, Median=7.0, Mean=8.0, Max=34


### `test_source3`

- **`business_name` Character Length:** Min=2, Median=25.0, Mean=25.7, Max=78
- **`business_name` Word Count:** Min=1, Median=4.0, Mean=3.6, Max=11
- **`business_address` Character Length:** Min=5, Median=44.0, Mean=50.1, Max=225
- **`business_address` Word Count:** Min=1, Median=7.0, Mean=7.7, Max=36


## 6. Sample Ground Truth Multi-Match Records

- **Source 1 ID:** `S1-965667` -> **Matches (5):** `S2-681193310, S2-743505751, S3-775321672, S3-11291185, S3-860443364`
- **Source 1 ID:** `S1-55344266` -> **Matches (4):** `S2-249013014, S2-197070651, S3-478195123, S3-384364074`
- **Source 1 ID:** `S1-343815751` -> **Matches (3):** `S2-790675320, S2-479876582, S3-878454467`
- **Source 1 ID:** `S1-656753428` -> **Matches (3):** `S2-153058913, S2-24659151, S3-679606215`
- **Source 1 ID:** `S1-102811957` -> **Matches (6):** `S2-478959098, S2-553508714, S2-625774905, S3-728090388, S3-928796641, S3-449308785`


## 7. Noise Patterns & Strategic Implications for Modeling

1. **Data Scale:** S1 has 2,206,821 rows, S2 has 5,034,616 rows, S3 has 5,285,603 rows. Total training candidate universe is ~2.2M x 10.3M pairs. Strict and efficient blocking is critical.
2. **Entity Resolution Topology:** S1 entities are canonical reference businesses. S2 and S3 are noisy crawls. One S1 entity can match 0, 1, or multiple entities across S2 and S3.
3. **Singleton Dominance & Precision:** Singletons (zero matches) are a major portion of S1 entities. Under Macro F0.5 ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$), Precision is weighted 4x more heavily than Recall ($(\beta=0.5)^2 = 0.25$). False merges severely degrade the score, so strict thresholding and confidence gating are essential.
4. **Country Generalization:** Country values in train vs test (e.g. France / `FR` present in test) mandate country-agnostic normalization (e.g. unicode diacritics stripping, robust legal suffix handling) and country-aware blocking.
5. **Address Variations:** Noise includes punctuation variations, street abbreviations, suite/floor omissions, postal code formatting shifts, and city/state reordering.