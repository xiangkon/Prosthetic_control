# TUM 各受试者各试次任务对照

来源：每名受试者各自的 `trial_info.mat`。中文名称为原始英文描述的翻译，未推断动作的具体关节方向。

trial 是采集试次编号，不是任务编号；不同受试者的同号 trial 可能属于不同任务。
数字列表逐个列出，不将中间跳号视为有效试次；“—”表示该模态未被元数据列出。

MoCap/EMG 使用 `trial_001.mat` 格式，EEG 使用 `trial_0001.mat` 格式。
磁盘上存在但未被引用的文件单独列出，不能仅凭文件存在就认定为有效标签试次。

## 运动学 MoCap 汇总

| Task | 任务 | subj_001 | subj_002 | subj_003 | subj_004 | subj_005 | subj_006 |
|---:|---|---|---|---|---|---|---|
| 0 | 初始放松姿态 | 001、002、003 | 001、002、003 | 001 | 003、004、005 | 002、003、004 | 001、003、004 |
| 1 | OK 手势 | 005、007 | 006、007、008 | 009 | 006、007、008 | 006、007、008 | 005、006、007 |
| 2 | 拇指向下 | 008、009、010 | 009、010、011 | 010、011 | 009、010、011 | 009、010、011 | 008、009、010 |
| 3 | 欢呼／庆祝手势 | 011、013 | 012、013、014 | 013、014、015 | 012、013、014 | 012、013、014 | 011、012、013 |
| 4 | 搭便车手势 | 014、015、016 | 015、016、017 | 016、018、019 | 015、016、017 | 015、016、017 | 014、015、016 |
| 5 | 用手遮挡面部阳光 | 017、018、019 | 019、020、021 | 021、022、023 | 018、019、020 | 018、019、020 | 017、018、019 |
| 6 | 打招呼 | 020、021、022 | 022、023、024 | 025、026 | 023、024、025 | 021、022、023 | 020、021、022 |
| 7 | 军礼 | 023、024、025 | 025、026、027 | 028、029、030 | 028、029、030 | 025、026 | 024、025 |
| 8 | 停止手势 | 026、027、028 | 028、029、030 | 031、032、033 | 031、032、033 | 027、028、029 | 026、027 |
| 9 | 指向手势 | 029、030、031 | 031、032、033 | 034、035、036 | 034、035、036 | 031、032 | 029、030、031 |
| 10 | 安静／嘘手势 | 032、033、034 | 036、037、038 | 038 | 037、038、039 | 033、034、035 | 032、033、034 |
| 11 | 伸手抓取小手提箱 | 035、036、038 | 040、041 | 042 | 041、042、043 | 036、037、038 | 035、036、037 |
| 12 | 伸手抓取玻璃杯 | 039、040、041 | 042、044、045 | 043、044、045 | 044、045、046 | 039、040、041 | 038、039、040 |
| 13 | 伸手抓取电话听筒 | 042、043、044 | 046、047、048 | 047、048、049 | 047、048、049 | 042、043、044 | 041、042、043 |
| 14 | 伸手抓取书 | 045、046、047 | 049、050、051 | 050、051、052 | 053、054、055 | 045、046、047 | 044、045、046 |
| 15 | 伸手抓取小杯子 | 048、049、050 | 052、053、054 | 053、054、055 | 056、057、058 | 048、049、050 | 047、048、049 |
| 16 | 伸手抓取苹果 | 051、052、053 | 055、056、057 | 056、057、058 | 059、060、061 | 051、052、053 | 050、051、052 |
| 17 | 伸手抓取帽子 | 054、055、056 | 058、059、060 | 059、060、063 | 062、063、064 | 054、055、056 | 053、054 |
| 18 | 伸手抓取杯子 | 057、058、059 | 061、062、063 | 064、065、066 | 065、066、067 | 057、058、059 | 056、057、058 |
| 19 | 接过他人递来的托盘 | 060、061、062 | 064、065、066 | 067、068、069 | 068、069、070 | 060、061、062 | 059、060、061 |
| 20 | 伸手抓取锁中的钥匙 | 063、064、065 | 069、070、071 | 070、071、072 | 071、072、073 | 063、064、065 | 062、064、065 |
| 21 | 伸手抓取瓶子 | 067、068、069 | 072、073、074 | 073、074、075 | 074、075、076 | 066、067、068 | 066、067 |
| 22 | 伸手抓取智能手机 | 070、071、072 | 075、076、077 | 078 | 077、078、079 | 069、070、071 | 069、070、071 |
| 23 | 伸手抓取牙刷 | 073、074、075 | 078、079、080 | 079、080、081 | 080、081、082 | 072、074 | 072、073、074 |
| 24 | 伸手抓取笔记本电脑 | 076、077、078 | 081、082、084 | 082、083、084 | 083、084、086 | 075、076、077 | 075、076、077 |
| 25 | 伸手抓取笔（pen） | 079、080、082 | 085、086、087 | 085、086、087 | 087、088、089 | 078、079、081 | 078、079、081 |
| 26 | 伸手抓取铅笔 | 083、084、085 | 088、089、090 | 089、090 | 090、091、092 | 082、083、084 | 082、083、084 |
| 27 | 伸手抓取茶包 | 086、087、088 | 091、092、093 | 091、092、093 | 093、094、095 | 085、086、087 | 085、086、087 |
| 28 | 伸手抓取门把手 | 090、091、092 | 094、095、096 | 094、095、096 | 096、097、099 | 088、089、090 | 089、090 |
| 29 | 伸手抓取网球 | 093、094、095 | 097、099 | 097、098、099 | 100、101、102 | 091、093、094 | 091、092、093 |
| 30 | 伸手抓取瓶盖 | 096、097、098 | 101、102 | 100、101、103 | 103、105、106 | 095、096、097 | 094、095、096 |
| 31 | 眼睛垂直运动 | — | — | — | — | — | — |
| 32 | 眼睛水平运动 | — | — | — | — | — | — |
| 33 | 睁眼／闭眼 | — | — | — | — | — | — |

## subj_001 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 001、002、003 | 001、002、003 | 001、002、003 |
| 1 | OK 手势 | Ok gesture | 005、007 | 005、006、007 | 005、006、007 |
| 2 | 拇指向下 | Thumb down | 008、009、010 | 008、009、010 | 008、009、010 |
| 3 | 欢呼／庆祝手势 | Exultation | 011、013 | 011、012、013 | 011、012、013 |
| 4 | 搭便车手势 | Hitchhiking | 014、015、016 | 014、015、016 | 014、015、016 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 017、018、019 | 017、018、019 | 017、018、019 |
| 6 | 打招呼 | Greet | 020、021、022 | 020、021、022 | 020、021、022 |
| 7 | 军礼 | Military salute | 023、024、025 | 023、024、025 | 023、024、025 |
| 8 | 停止手势 | Stop gesture | 026、027、028 | 026、027、028 | 026、027、028 |
| 9 | 指向手势 | Pointing | 029、030、031 | 029、030、031 | 029、030、031 |
| 10 | 安静／嘘手势 | Silence gesture | 032、033、034 | 032、033、034 | 032、033、034 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 035、036、038 | 035、036、038 | 035、036、038 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 039、040、041 | 039、040、041 | 039、040、041 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 042、043、044 | 042、043、044 | 042、043、044 |
| 14 | 伸手抓取书 | Reach and grasp a book | 045、046、047 | 045、046、047 | 045、046、047 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 048、049、050 | 048、049、050 | 048、049、050 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 051、052、053 | 051、052、053 | 051、052、053 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 054、055、056 | 054、055、056 | 054、055、056 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 057、058、059 | 057、058、059 | 057、058、059 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 060、061、062 | 060、061、062 | 060、061、062 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 063、064、065 | 063、064、065 | 063、064、065 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 067、068、069 | 067、068、069 | 067、068、069 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 070、071、072 | 070、071、072 | 070、071、072 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 073、074、075 | 073、074、075 | 073、074、075 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 076、077、078 | 076、077、078 | 076、077、078 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 079、080、082 | 079、080、082 | 079、080、082 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 083、084、085 | 083、084、085 | 083、084、085 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 086、087、088 | 086、087、088 | 086、087、088 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 090、091、092 | 090、091、092 | 090、091、092 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 093、094、095 | 093、094、095 | 093、094、095 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 096、097、098 | 096、097、098 | 096、097、098 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 111、113 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 114、115、116 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 117、118、119 |

## subj_002 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 001、002、003 | 001、002 | 001、002、003 |
| 1 | OK 手势 | Ok gesture | 006、007、008 | 006、007、008 | 006、007、008 |
| 2 | 拇指向下 | Thumb down | 009、010、011 | 009、010、011 | 009、010、011 |
| 3 | 欢呼／庆祝手势 | Exultation | 012、013、014 | 012、013、014 | 012、013、014 |
| 4 | 搭便车手势 | Hitchhiking | 015、016、017 | 015、016、017 | 015、016、017 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 019、020、021 | 019、020、021 | 019、020、021 |
| 6 | 打招呼 | Greet | 022、023、024 | 022、023、024 | 022、023、024 |
| 7 | 军礼 | Military salute | 025、026、027 | 025、026、027 | 025、026、027 |
| 8 | 停止手势 | Stop gesture | 028、029、030 | 028、029、030 | 028、029、030 |
| 9 | 指向手势 | Pointing | 031、032、033 | 031、032、033 | 031、032、033 |
| 10 | 安静／嘘手势 | Silence gesture | 036、037、038 | 036、037、038 | 036、037、038 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 040、041 | 039、040 | 039、040、041 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 042、044、045 | 044、045 | 042、044、045 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 046、047、048 | 046、047、048 | 046、047、048 |
| 14 | 伸手抓取书 | Reach and grasp a book | 049、050、051 | 049、050、051 | 049、050、051 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 052、053、054 | 052、053、054 | 052、053、054 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 055、056、057 | 055、056、057 | 055、056、057 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 058、059、060 | 058、059、060 | 058、059、060 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 061、062、063 | 061、062、063 | 061、062、063 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 064、065、066 | 064、065、066 | 064、065、066 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 069、070、071 | 069、070、071 | 069、070、071 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 072、073、074 | 072、073、074 | 072、073、074 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 075、076、077 | 075、076、077 | 075、076、077 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 078、079、080 | 078、079、080 | 078、079、080 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 081、082、084 | 081、082、084 | 081、082、084 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 085、086、087 | 085、086、087 | 085、086、087 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 088、089、090 | 088、089 | 088、089、090 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 091、092、093 | 091、092、093 | 091、092、093 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 094、095、096 | 094、095、096 | 094、095、096 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 097、099 | 097、098、099 | 097、098、099 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 101、102 | 100、101、102 | 100、101、102 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 115、117、118 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 116、119、120 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 121、122、123 |

## subj_003 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 001 | 001、002、003 | 001、002、003 |
| 1 | OK 手势 | Ok gesture | 009 | 005、007、009 | 005、007、009 |
| 2 | 拇指向下 | Thumb down | 010、011 | 010、011、012 | 010、011、012 |
| 3 | 欢呼／庆祝手势 | Exultation | 013、014、015 | 013、014、015 | 013、014、015 |
| 4 | 搭便车手势 | Hitchhiking | 016、018、019 | 016、018、019 | 016、018、019 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 021、022、023 | 021、022、023 | 021、022、023 |
| 6 | 打招呼 | Greet | 025、026 | 025、026、027 | 025、026、027 |
| 7 | 军礼 | Military salute | 028、029、030 | 028、029、030 | 028、029、030 |
| 8 | 停止手势 | Stop gesture | 031、032、033 | 031、032、033 | 031、032、033 |
| 9 | 指向手势 | Pointing | 034、035、036 | 034、035、036 | 034、035、036 |
| 10 | 安静／嘘手势 | Silence gesture | 038 | 037、038、039 | 037、038、039 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 042 | 040、041、042 | 040、041、042 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 043、044、045 | 043、044、045 | 043、044、045 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 047、048、049 | 047、048、049 | 047、048、049 |
| 14 | 伸手抓取书 | Reach and grasp a book | 050、051、052 | 050、051、052 | 050、051、052 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 053、054、055 | 053、054、055 | 053、054、055 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 056、057、058 | 056、057、058 | 056、057、058 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 059、060、063 | 059、060、063 | 059、060、063 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 064、065、066 | 064、065、066 | 064、065、066 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 067、068、069 | 067、068、069 | 067、068、069 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 070、071、072 | 070、071、072 | 070、071、072 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 073、074、075 | 073、074、075 | 073、074、075 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 078 | 076、077、078 | 076、077、078 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 079、080、081 | 079、080、081 | 079、080、081 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 082、083、084 | 082、083、084 | 082、083、084 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 085、086、087 | 085、086、087 | 085、086、087 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 089、090 | 088、089、090 | 088、089、090 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 091、092、093 | 091、092、093 | 091、092、093 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 094、095、096 | 094、095、096 | 094、095、096 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 097、098、099 | 097、098、099 | 097、098、099 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 100、101、103 | 100、101、103 | 100、101、103 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 116、117、118 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 119、120、121 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 122、123、124 |

## subj_004 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 003、004、005 | 003、004、005 | 003、004、005 |
| 1 | OK 手势 | Ok gesture | 006、007、008 | 006、007、008 | 006、007、008 |
| 2 | 拇指向下 | Thumb down | 009、010、011 | 009、010、011 | 009、010、011 |
| 3 | 欢呼／庆祝手势 | Exultation | 012、013、014 | 012、013、014 | 012、013、014 |
| 4 | 搭便车手势 | Hitchhiking | 015、016、017 | 015、016、017 | 015、016、017 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 018、019、020 | 018、019、020 | 018、019、020 |
| 6 | 打招呼 | Greet | 023、024、025 | 023、024、025 | 023、024、025 |
| 7 | 军礼 | Military salute | 028、029、030 | 028、029、030 | 028、029、030 |
| 8 | 停止手势 | Stop gesture | 031、032、033 | 031、032、033 | 031、032、033 |
| 9 | 指向手势 | Pointing | 034、035、036 | 034、035、036 | 034、035、036 |
| 10 | 安静／嘘手势 | Silence gesture | 037、038、039 | 037、038、039 | 037、038、039 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 041、042、043 | 041、042、043 | 041、042、043 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 044、045、046 | 044、045、046 | 044、045、046 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 047、048、049 | 047、048、049 | 047、048、049 |
| 14 | 伸手抓取书 | Reach and grasp a book | 053、054、055 | 053、054、055 | 053、054、055 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 056、057、058 | 056、057、058 | 056、057、058 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 059、060、061 | 059、060、061 | 059、060、061 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 062、063、064 | 062、063、064 | 062、063、064 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 065、066、067 | 065、066、067 | 065、066、067 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 068、069、070 | 068、069、070 | 068、069、070 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 071、072、073 | 071、072、073 | 071、072、073 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 074、075、076 | 074、075、076 | 074、075、076 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 077、078、079 | 077、078、079 | 077、078、079 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 080、081、082 | 080、081、082 | 080、081、082 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 083、084、086 | 083、084、086 | 083、084、086 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 087、088、089 | 087、088、089 | 087、088、089 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 090、091、092 | 090、091、092 | 090、091、092 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 093、094、095 | 093、094、095 | 093、094、095 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 096、097、099 | 096、097、099 | 096、097、099 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 100、101、102 | 100、101、102 | 100、101、102 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 103、105、106 | 103、105、106 | 103、105、106 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 121、122、123 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 124、125、126 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 127、128、129 |

## subj_005 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 002、003、004 | 002、003、004 | 002、003、004 |
| 1 | OK 手势 | Ok gesture | 006、007、008 | 006、007、008 | 006、007、008 |
| 2 | 拇指向下 | Thumb down | 009、010、011 | 009、010、011 | 009、010、011 |
| 3 | 欢呼／庆祝手势 | Exultation | 012、013、014 | 012、013、014 | 012、013、014 |
| 4 | 搭便车手势 | Hitchhiking | 015、016、017 | 015、016、017 | 015、016、017 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 018、019、020 | 018、019、020 | 018、019、020 |
| 6 | 打招呼 | Greet | 021、022、023 | 021、022、023 | 021、022、023 |
| 7 | 军礼 | Military salute | 025、026 | 024、025、026 | 024、025、026 |
| 8 | 停止手势 | Stop gesture | 027、028、029 | 027、028、029 | 027、028、029 |
| 9 | 指向手势 | Pointing | 031、032 | 030、031、032 | 030、031、032 |
| 10 | 安静／嘘手势 | Silence gesture | 033、034、035 | 033、034、035 | 033、034、035 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 036、037、038 | 036、037、038 | 036、037、038 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 039、040、041 | 039、040、041 | 039、040、041 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 042、043、044 | 042、043、044 | 042、043、044 |
| 14 | 伸手抓取书 | Reach and grasp a book | 045、046、047 | 045、046、047 | 045、046、047 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 048、049、050 | 048、049、050 | 048、049、050 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 051、052、053 | 051、052、053 | 051、052、053 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 054、055、056 | 054、055、056 | 054、055、056 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 057、058、059 | 057、058、059 | 057、058、059 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 060、061、062 | 060、061、062 | 060、061、062 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 063、064、065 | 063、064、065 | 063、064、065 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 066、067、068 | 066、067、068 | 066、067、068 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 069、070、071 | 069、070、071 | 069、070、071 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 072、074 | 072、073、074 | 072、073、074 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 075、076、077 | 075、076、077 | 075、076、077 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 078、079、081 | 078、079、081 | 078、079、081 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 082、083、084 | 082、083、084 | 082、083、084 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 085、086、087 | 085、086、087 | 085、086、087 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 088、089、090 | 088、089、090 | 088、089、090 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 091、093、094 | 091、093、094 | 091、093、094 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 095、096、097 | 095、096、097 | 095、096、097 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 110、111、112 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 113、114、115 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 116、117、118 |

## subj_006 全部模态

| Task | 中文任务 | 原始英文描述 | MoCap trial | EMG trial | EEG trial |
|---:|---|---|---|---|---|
| 0 | 初始放松姿态 | Initinal & relaxed pose | 001、003、004 | 001、003、004 | 001、003、004 |
| 1 | OK 手势 | Ok gesture | 005、006、007 | 005、006、007 | 005、006、007 |
| 2 | 拇指向下 | Thumb down | 008、009、010 | 008、009、010 | 008、009、010 |
| 3 | 欢呼／庆祝手势 | Exultation | 011、012、013 | 011、012、013 | 011、012、013 |
| 4 | 搭便车手势 | Hitchhiking | 014、015、016 | 014、015、016 | 014、015、016 |
| 5 | 用手遮挡面部阳光 | Block out sun from own face | 017、018、019 | 017、018、019 | 017、018、019 |
| 6 | 打招呼 | Greet | 020、021、022 | 020、021、022 | 020、021、022 |
| 7 | 军礼 | Military salute | 024、025 | 023、024、025 | 023、024、025 |
| 8 | 停止手势 | Stop gesture | 026、027 | 026、027、028 | 026、027、028 |
| 9 | 指向手势 | Pointing | 029、030、031 | 029、030、031 | 029、030、031 |
| 10 | 安静／嘘手势 | Silence gesture | 032、033、034 | 032、033、034 | 032、033、034 |
| 11 | 伸手抓取小手提箱 | Reach and grasp a small suitcase | 035、036、037 | 035、036、037 | 035、036、037 |
| 12 | 伸手抓取玻璃杯 | Reach and grasp a glass | 038、039、040 | 038、039、040 | 038、039、040 |
| 13 | 伸手抓取电话听筒 | Reach and grasp a phone receiver | 041、042、043 | 041、042、043 | 041、042、043 |
| 14 | 伸手抓取书 | Reach and grasp a book | 044、045、046 | 044、045、046 | 044、045、046 |
| 15 | 伸手抓取小杯子 | Reach and grasp a small cup | 047、048、049 | 047、048、049 | 047、048、049 |
| 16 | 伸手抓取苹果 | Reach and grasp a an apple | 050、051、052 | 050、051、052 | 050、051、052 |
| 17 | 伸手抓取帽子 | Reach and grasp a hat | 053、054 | 053、054、055 | 053、054、055 |
| 18 | 伸手抓取杯子 | Reach and grasp a cup | 056、057、058 | 056、057、058 | 056、057、058 |
| 19 | 接过他人递来的托盘 | Receive a tray from someone | 059、060、061 | 059、060、061 | 059、060、061 |
| 20 | 伸手抓取锁中的钥匙 | Reach and grasp a key in a lock | 062、064、065 | 062、064、065 | 062、064、065 |
| 21 | 伸手抓取瓶子 | Reach and grasp a bottle | 066、067 | 066、067、068 | 066、067、068 |
| 22 | 伸手抓取智能手机 | Reach and grasp a smatphone | 069、070、071 | 069、070、071 | 069、070、071 |
| 23 | 伸手抓取牙刷 | Reach and grasp a tothbrush | 072、073、074 | 072、073、074 | 072、073、074 |
| 24 | 伸手抓取笔记本电脑 | Reach and grasp a laptop | 075、076、077 | 075、076、077 | 075、076、077 |
| 25 | 伸手抓取笔（pen） | Reach and grasp a pen | 078、079、081 | 078、079、081 | 078、079、081 |
| 26 | 伸手抓取铅笔 | Reach and grasp a pencil | 082、083、084 | 082、083、084 | 082、083、084 |
| 27 | 伸手抓取茶包 | Reach and grasp a tea bag | 085、086、087 | 085、086、087 | 085、086、087 |
| 28 | 伸手抓取门把手 | Reach and grasp a doorknob | 089、090 | 088、089、090 | 088、089、090 |
| 29 | 伸手抓取网球 | Reach and grasp a tennis ball | 091、092、093 | 091、092、093 | 091、092、093 |
| 30 | 伸手抓取瓶盖 | Reach and grasp a cap of a bottle | 094、095、096 | 094、095、096 | 094、095、096 |
| 31 | 眼睛垂直运动 | Vertical movement of eyes | — | — | 110、111、112 |
| 32 | 眼睛水平运动 | Horizontal movement of eyes | — | — | 113、114、115 |
| 33 | 睁眼／闭眼 | Open/close of eyes | — | — | 116、117、118 |

## 磁盘上未被 trial_info 引用的文件

这里不自动赋予任务，也不自动删除。无下划线的 trial001.mat 是另一套文件名，不能与 trial_001.mat 混为同一个标签引用。

**subj_001/MoCap**：`trial_004.mat`、`trial_006.mat`、`trial_012.mat`、`trial_037.mat`、`trial_066.mat`、`trial_081.mat`、`trial_089.mat`。

**subj_001/EMG**：`trial_004.mat`、`trial_037.mat`、`trial_066.mat`、`trial_081.mat`、`trial_089.mat`。

**subj_001/EEG**：`trial_0120.mat`。

**subj_002/MoCap**：`trial_004.mat`、`trial_005.mat`、`trial_018.mat`、`trial_034.mat`、`trial_035.mat`、`trial_039.mat`、`trial_043.mat`、`trial_067.mat`、`trial_068.mat`、`trial_083.mat`、`trial_098.mat`、`trial_100.mat`。

**subj_002/EMG**：`trial_003.mat`、`trial_004.mat`、`trial_005.mat`、`trial_018.mat`、`trial_034.mat`、`trial_035.mat`、`trial_041.mat`、`trial_042.mat`、`trial_043.mat`、`trial_067.mat`、`trial_068.mat`、`trial_083.mat`、`trial_090.mat`。

**subj_002/EEG**：无。

**subj_003/MoCap**：`trial_002.mat`、`trial_003.mat`、`trial_004.mat`、`trial_006.mat`、`trial_007.mat`、`trial_008.mat`、`trial_012.mat`、`trial_017.mat`、`trial_020.mat`、`trial_024.mat`、`trial_027.mat`、`trial_037.mat`、`trial_039.mat`、`trial_040.mat`、`trial_041.mat`、`trial_046.mat`、`trial_061.mat`、`trial_062.mat`、`trial_076.mat`、`trial_077.mat`、`trial_088.mat`、`trial_102.mat`。

**subj_003/EMG**：`trial_004.mat`、`trial_006.mat`、`trial_008.mat`、`trial_017.mat`、`trial_020.mat`、`trial_024.mat`、`trial_046.mat`、`trial_061.mat`、`trial_062.mat`、`trial_102.mat`。

**subj_003/EEG**：无。

**subj_004/MoCap**：`trial_001.mat`、`trial_002.mat`、`trial_021.mat`、`trial_022.mat`、`trial_026.mat`、`trial_027.mat`、`trial_040.mat`、`trial_050.mat`、`trial_051.mat`、`trial_052.mat`、`trial_085.mat`、`trial_098.mat`、`trial_104.mat`。

**subj_004/EMG**：`trial_001.mat`、`trial_002.mat`、`trial_021.mat`、`trial_022.mat`、`trial_026.mat`、`trial_027.mat`、`trial_040.mat`、`trial_050.mat`、`trial_051.mat`、`trial_052.mat`、`trial_085.mat`、`trial_098.mat`、`trial_104.mat`。

**subj_004/EEG**：无。

**subj_005/MoCap**：`trial001.mat`、`trial002.mat`、`trial003.mat`、`trial004.mat`、`trial005.mat`、`trial006.mat`、`trial007.mat`、`trial008.mat`、`trial009.mat`、`trial010.mat`、`trial011.mat`、`trial012.mat`、`trial013.mat`、`trial014.mat`、`trial015.mat`、`trial016.mat`、`trial017.mat`、`trial018.mat`、`trial019.mat`、`trial020.mat`、`trial021.mat`、`trial022.mat`、`trial023.mat`、`trial024.mat`、`trial025.mat`、`trial026.mat`、`trial027.mat`、`trial028.mat`、`trial029.mat`、`trial030.mat`、`trial031.mat`、`trial032.mat`、`trial033.mat`、`trial034.mat`、`trial035.mat`、`trial036.mat`、`trial037.mat`、`trial038.mat`、`trial039.mat`、`trial040.mat`、`trial041.mat`、`trial042.mat`、`trial043.mat`、`trial044.mat`、`trial045.mat`、`trial046.mat`、`trial047.mat`、`trial048.mat`、`trial049.mat`、`trial050.mat`、`trial051.mat`、`trial052.mat`、`trial053.mat`、`trial054.mat`、`trial055.mat`、`trial056.mat`、`trial057.mat`、`trial058.mat`、`trial059.mat`、`trial060.mat`、`trial061.mat`、`trial062.mat`、`trial063.mat`、`trial064.mat`、`trial065.mat`、`trial066.mat`、`trial067.mat`、`trial068.mat`、`trial069.mat`、`trial070.mat`、`trial071.mat`、`trial072.mat`、`trial073.mat`、`trial074.mat`、`trial075.mat`、`trial076.mat`、`trial077.mat`、`trial078.mat`、`trial079.mat`、`trial080.mat`、`trial081.mat`、`trial082.mat`、`trial083.mat`、`trial084.mat`、`trial085.mat`、`trial086.mat`、`trial087.mat`、`trial088.mat`、`trial089.mat`、`trial090.mat`、`trial091.mat`、`trial092.mat`、`trial093.mat`、`trial094.mat`、`trial095.mat`、`trial096.mat`、`trial097.mat`、`trial098.mat`、`trial099.mat`、`trial100.mat`、`trial101.mat`、`trial102.mat`、`trial103.mat`、`trial104.mat`、`trial105.mat`、`trial106.mat`、`trial107.mat`、`trial108.mat`、`trial109.mat`、`trial_001.mat`、`trial_005.mat`、`trial_024.mat`、`trial_030.mat`、`trial_073.mat`、`trial_080.mat`、`trial_092.mat`。

**subj_005/EMG**：`trial_001.mat`、`trial_005.mat`、`trial_080.mat`、`trial_092.mat`。

**subj_005/EEG**：无。

**subj_006/MoCap**：`trial001.mat`、`trial002.mat`、`trial003.mat`、`trial004.mat`、`trial005.mat`、`trial006.mat`、`trial007.mat`、`trial008.mat`、`trial009.mat`、`trial010.mat`、`trial011.mat`、`trial012.mat`、`trial013.mat`、`trial014.mat`、`trial015.mat`、`trial016.mat`、`trial017.mat`、`trial018.mat`、`trial019.mat`、`trial020.mat`、`trial021.mat`、`trial022.mat`、`trial023.mat`、`trial024.mat`、`trial025.mat`、`trial026.mat`、`trial027.mat`、`trial028.mat`、`trial029.mat`、`trial030.mat`、`trial031.mat`、`trial032.mat`、`trial033.mat`、`trial034.mat`、`trial035.mat`、`trial036.mat`、`trial037.mat`、`trial038.mat`、`trial039.mat`、`trial040.mat`、`trial041.mat`、`trial042.mat`、`trial043.mat`、`trial044.mat`、`trial045.mat`、`trial046.mat`、`trial047.mat`、`trial048.mat`、`trial049.mat`、`trial050.mat`、`trial051.mat`、`trial052.mat`、`trial053.mat`、`trial054.mat`、`trial055.mat`、`trial056.mat`、`trial057.mat`、`trial058.mat`、`trial059.mat`、`trial060.mat`、`trial061.mat`、`trial062.mat`、`trial063.mat`、`trial064.mat`、`trial065.mat`、`trial066.mat`、`trial067.mat`、`trial068.mat`、`trial069.mat`、`trial070.mat`、`trial071.mat`、`trial072.mat`、`trial073.mat`、`trial074.mat`、`trial075.mat`、`trial076.mat`、`trial077.mat`、`trial078.mat`、`trial079.mat`、`trial080.mat`、`trial081.mat`、`trial082.mat`、`trial083.mat`、`trial084.mat`、`trial085.mat`、`trial086.mat`、`trial087.mat`、`trial088.mat`、`trial089.mat`、`trial090.mat`、`trial091.mat`、`trial092.mat`、`trial093.mat`、`trial094.mat`、`trial095.mat`、`trial096.mat`、`trial097.mat`、`trial098.mat`、`trial099.mat`、`trial100.mat`、`trial101.mat`、`trial102.mat`、`trial103.mat`、`trial104.mat`、`trial105.mat`、`trial106.mat`、`trial107.mat`、`trial108.mat`、`trial109.mat`、`trial_002.mat`、`trial_023.mat`、`trial_028.mat`、`trial_055.mat`、`trial_063.mat`、`trial_068.mat`、`trial_080.mat`、`trial_088.mat`。

**subj_006/EMG**：`trial_002.mat`、`trial_063.mat`、`trial_080.mat`。

**subj_006/EEG**：无。
