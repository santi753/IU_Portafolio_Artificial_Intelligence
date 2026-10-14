# Phase 1 Feasibility Report

<!-- BEGIN dataset -->
## 1. Dataset check

Run: 2026-10-14 15:43:43 | Python 3.12.10 | pandas 3.0.6

Files in data/raw/: all-data.csv (672,006 B), License.txt (391 B), README.txt (3,967 B), Sentences_50Agree.txt (671,355 B), Sentences_66Agree.txt (580,254 B), Sentences_75Agree.txt (466,991 B), Sentences_AllAgree.txt (299,636 B)

### all-data.csv

- Encoding that worked: **latin-1** (UTF-8 tried first)
- Header row detected: **False**; columns named `label`, `text`
- Rows: **4846**
- Missing values: label 0, text 0
- Duplicate rows (label+text): 6

| label | count | percent |
|---|---|---|
| neutral | 2879 | 59.4% |
| positive | 1363 | 28.1% |
| negative | 604 | 12.5% |

Distinct labels: 3 -> at least three labels: **True**

Text length in words: min 2, median 21, mean 23.1, max 81

#### Duplicated texts

Distinct texts occurring more than once: 8 (16 rows); with conflicting labels: **2**

| text (first 90 chars) | occurrences | labels | conflict |
|---|---|---|---|
| Ahlstrom 's share is quoted on the NASDAQ OMX Helsinki . | 2 | neutral | no |
| Proha Plc ( Euronext :7327 ) announced today ( 19 May ) that its fully-owned subsidiary ,  | 2 | neutral | no |
| SSH Communications Security Corporation is headquartered in Helsinki , Finland . | 2 | neutral | no |
| TELECOMWORLDWIRE-7 April 2006-TJ Group Plc sells stake in Morning Digital Design Oy Finnis | 2 | neutral, positive | **YES** |
| The Group 's business is balanced by its broad portfolio of sports and presence in all maj | 2 | neutral, positive | **YES** |
| The company serves customers in various industries , including process and resources , ind | 2 | neutral | no |
| The issuer is solely responsible for the content of this announcement . | 2 | neutral | no |
| The report profiles 614 companies including many key and niche players worldwide such as B | 2 | neutral | no |

#### Examples (2 per label)

| label | text |
|---|---|
| negative | The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the compa |
| negative | A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertise |
| neutral | According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing . |
| neutral | Technopolis plans to develop in stages an area of no less than 100,000 square meters in order to host companies working in computer technologies and t |
| positive | With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materia |
| positive | According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an |

### Agreement-level .txt files (Latin-1, split on last '@')

| file | sentences | negative | neutral | positive |
|---|---|---|---|---|
| Sentences_50Agree.txt | 4846 | 604 (12.5%) | 2879 (59.4%) | 1363 (28.1%) |
| Sentences_66Agree.txt | 4217 | 514 (12.2%) | 2535 (60.1%) | 1168 (27.7%) |
| Sentences_75Agree.txt | 3453 | 420 (12.2%) | 2146 (62.1%) | 887 (25.7%) |
| Sentences_AllAgree.txt | 2264 | 303 (13.4%) | 1391 (61.4%) | 570 (25.2%) |

### all-data.csv vs Sentences_50Agree.txt

- Rows matching on label+text: **4735 / 4846**
- Mismatching rows: 111; of these, match after mapping non-ASCII characters to '_': 111
- Labels identical row by row (same order, same length): True

| row | label csv/txt | csv snippet | txt snippet |
|---|---|---|---|
| 14 | positive/positive | "othing retail chain Sepp+\xc6l+\xc6 's sales i" | "othing retail chain Sepp+\xf1l+\xf1 's sales i" |
| 30 | positive/positive | 'irector , Vladimir Brand+_o , as saying ' | 'irector , Vladimir Brand+\xfao , as saying ' |

### Observations

- all-data.csv decodes as latin-1, has no header row, 4846 rows and 3 labels (neutral 2879 (59.4%); positive 1363 (28.1%); negative 604 (12.5%)).
- 6 exact duplicate rows; 8 texts occur more than once, 2 of them with conflicting labels.
- 4735 of 4846 rows match Sentences_50Agree.txt on label+text; 111 of the 111 mismatches match once non-ASCII characters are mapped to '_'; labels identical row by row: True.
- Sentence counts per agreement file: Sentences_50Agree.txt 4846, Sentences_66Agree.txt 4217, Sentences_75Agree.txt 3453, Sentences_AllAgree.txt 2264.
<!-- END dataset -->

<!-- BEGIN news -->
## 2. News feed check (Google News RSS)

Run: 2026-10-14 15:43:44 | Python 3.12.10 | requests 2.34.2 | feedparser 6.0.14

URL template: `https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en`

#### Query `Tesla when:7d`

- HTTP status: 200
- Items: **100**
- Published range: 2026-10-07 14:01 to 2026-10-14 11:30 UTC
- Title contains 'Tesla' (case-insensitive): 97 (97%)
- Title ends with ' - <source name>': 100 (100%)
- Summary contains text beyond title/source (HTML stripped): 3 (3%)
- Fields (items having it): `title` 100, `title_detail` 100, `links` 100, `link` 100, `id` 100, `guidislink` 100, `published` 100, `published_parsed` 100, `summary` 100, `summary_detail` 100, `source` 100

| # | title | source | published |
|---|---|---|---|
| 1 | Tesla's China-made EV sales quicken in September - Reuters | Reuters | Fri, 09 Oct 2026 10:17:44 GMT |
| 2 | Tesla drops 'Full Self-Driving' brand name in Europe after regulator pushback - CNBC | CNBC | Fri, 09 Oct 2026 17:10:13 GMT |
| 3 | They Paid $50,000 and Waited Nine Years. Will Tesla’s Roadster Live Up to the Hype? - WSJ | WSJ | Sat, 10 Oct 2026 07:00:00 GMT |
| 4 | Tesla’s Roadster 2 Is Late—and Entering a Different EV World - WIRED | WIRED | Wed, 14 Oct 2026 09:15:00 GMT |
| 5 | Some Tesla Model 3s and Model Ys Can Now Power Some Homes with New V2H Bidirectional Charging - Car and Driver | Car and Driver | Thu, 08 Oct 2026 13:10:51 GMT |

Summary of item 1, HTML stripped: "Tesla's China-made EV sales quicken in September Reuters"

#### Query `Airbus when:7d`

- HTTP status: 200
- Items: **100**
- Published range: 2026-10-07 14:11 to 2026-10-14 13:03 UTC
- Title contains 'Airbus' (case-insensitive): 94 (94%)
- Title ends with ' - <source name>': 100 (100%)
- Summary contains text beyond title/source (HTML stripped): 0 (0%)
- Fields (items having it): `title` 100, `title_detail` 100, `links` 100, `link` 100, `id` 100, `guidislink` 100, `published` 100, `published_parsed` 100, `summary` 100, `summary_detail` 100, `source` 100

| # | title | source | published |
|---|---|---|---|
| 1 | Airbus U740 Valkyrie uncrewed combat aircraft takes off for its maiden flight in Europe - Airbus | Airbus | Mon, 12 Oct 2026 11:00:17 GMT |
| 2 | Airbus tests Valkyrie drone in Europe, plans 2029 combat readiness - Reuters | Reuters | Mon, 12 Oct 2026 09:05:06 GMT |
| 3 | British Airways is launching the world's largest business-class cabin on its Airbus A380s. Take a look inside. - Business Insider | Business Insider | Sun, 11 Oct 2026 10:58:00 GMT |
| 4 | FAA Orders Airbus A380 Engine Replacements After Uncontained Failure Forces Aborted Takeoff - Simple Flying | Simple Flying | Mon, 12 Oct 2026 21:54:23 GMT |
| 5 | Airbus SE (EADSY) Stock Price, News, Quote & History - Yahoo! Finance Canada | Yahoo! Finance Canada | Tue, 13 Oct 2026 15:36:36 GMT |

Summary of item 1, HTML stripped: "Airbus U740 Valkyrie uncrewed combat aircraft takes off for its maiden flight in Europe Airbus"

#### Query `Siemens when:7d`

- HTTP status: 200
- Items: **100**
- Published range: 2026-10-07 13:47 to 2026-10-14 13:32 UTC
- Title contains 'Siemens' (case-insensitive): 99 (99%)
- Title ends with ' - <source name>': 100 (100%)
- Summary contains text beyond title/source (HTML stripped): 0 (0%)
- Fields (items having it): `title` 100, `title_detail` 100, `links` 100, `link` 100, `id` 100, `guidislink` 100, `published` 100, `published_parsed` 100, `summary` 100, `summary_detail` 100, `source` 100

| # | title | source | published |
|---|---|---|---|
| 1 | Siemens Healthineers Sharpens Ultrasound to Focus on the Future of Procedural Imaging - Siemens Healthineers | Siemens Healthineers | Mon, 12 Oct 2026 16:11:37 GMT |
| 2 | Siemens Aktiengesellschaft (SIE.DE) Stock Price, News, Quote & History - Yahoo! Finance Canada | Yahoo! Finance Canada | Mon, 12 Oct 2026 16:27:32 GMT |
| 3 | Siemens focuses ultrasound business on procedural imaging - AuntMinnie | AuntMinnie | Tue, 13 Oct 2026 02:13:37 GMT |
| 4 | Siemens invests in Skeleton Technologies expanding its data center portfolio - Siemens press | Siemens press | Wed, 14 Oct 2026 10:17:22 GMT |
| 5 | Siemens Healthineers exits general ultrasound to focus on cardiac imaging - MassDevice | MassDevice | Tue, 13 Oct 2026 21:00:46 GMT |

Summary of item 1, HTML stripped: "Siemens Healthineers Sharpens Ultrasound to Focus on the Future of Procedural Imaging Siemens Healthineers"

#### Query `Tesla` (without when:7d)

- HTTP status: 200
- Items: **102**
- Oldest published: 2026-10-08 13:10 UTC

### Observations

- `Tesla when:7d`: 100 items, published 2026-10-07 14:01 to 2026-10-14 11:30 UTC; keyword in title 97 (97%); source suffix 100 (100%); summary with extra text 3 (3%).
- `Airbus when:7d`: 100 items, published 2026-10-07 14:11 to 2026-10-14 13:03 UTC; keyword in title 94 (94%); source suffix 100 (100%); summary with extra text 0 (0%).
- `Siemens when:7d`: 100 items, published 2026-10-07 13:47 to 2026-10-14 13:32 UTC; keyword in title 99 (99%); source suffix 100 (100%); summary with extra text 0 (0%).
- `Tesla` without when:7d: 102 items, oldest 2026-10-08 13:10 UTC.
<!-- END news -->
