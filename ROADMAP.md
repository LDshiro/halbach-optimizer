
---

## ROADMAP.md

```markdown
# ROADMAP.md
鏈儣銉偢銈с偗銉堛伄銉儠銈°偗銈裤儶銉炽偘瑷堢敾锛圕odex銇浮銇欎綔妤▓鐢绘浉锛夈仹銇欍��
鐩殑锛歚robust_opt_halbach_gradnorm_minimal.py` 銈掍繚瀹堟�с伄楂樸亜妲嬮�犮伕鍒嗗壊銇椼�佸幊鏍笺仾鍨嬨儊銈с儍銈仺銉嗐偣銉堛仹瀹夊叏銇敼鑹仹銇嶃倠銈堛亞銇仚銈嬨��

## 鍓嶆彁
- Windows + PowerShell
- Python 3.11 + venv
- GitHub銇ф柊瑕忋儶銉濅綔鎴愶紙绗�1绔犮仹瀹熸柦锛�
- CI銇皫鍏ャ仐銇亜锛堛儹銉笺偒銉仹pytest/format/typecheck銈掑洖銇欙級
- 鍨嬨儊銈с儍銈伅寮枫倎锛坢ypy strict锛�
- License: MIT
- 閰嶅竷浜堝畾銇仐锛坧ip package鍖栥伅銇椼仾銇勶級
- ROI涓婇檺銇洰瀹夛細鍐嗙瓛鍐呭伌鐩村緞銇磩80%绋嬪害锛堛仧銇犮仐ROI銇紩鏁般仹鍙锛�

---

## Milestones锛堢珷绔嬨仸锛�
### 0) Agent docs锛堝畬浜嗘潯浠讹級
- [ ] AGENTS.md / WORKFLOW.md / ROADMAP.md 銈掕拷鍔�
- [ ] 鈥滀笉澶夋潯浠讹紙鐩殑闁㈡暟/淇濆瓨浜掓彌锛夆�� 銈掓槑鏂囧寲

### 1) Repo bootstrap锛堛儹銉笺偒銉啋GitHub锛�
- [ ] venv锛�3.11锛夋墜闋嗐倰README銇浉銇�
- [ ] MIT LICENSE 銈掕拷鍔�
- [ ] .gitignore 銈掓暣鍌�
- [ ] 銉欍兗銈广偝銉笺儔銈� `src/` or 銉兗銉堛伀閰嶇疆銇椼�佸垵鍥炪偪銈� `v0-baseline`

### 2) Tooling锛坒ormatter/lint/typecheck锛�
- [ ] black / ruff / isort / mypy(strict) / pytest 銈掑皫鍏ワ紙pyproject.toml锛�
- [ ] 銉兗銈儷瀹熻鎵嬮爢銈扺ORKFLOW銇歌拷瑷�
- [ ] 锛堜换鎰忥級pre-commit灏庡叆锛圕I銇仐銇с倐鍝佽唱缍寔锛�

### 3) Tests锛堟渶灏忓畨鍏ㄧ恫锛�
- [ ] 銈广儮銉笺偗銉嗐偣銉堬紙灏廟OI銇� objective 銇屽嫊銇忥級
- [ ] gradcheck锛堣В鏋愬嬀閰� vs FD锛�1e-6銆�1e-9绱氥仹涓�鑷达級
- [ ] 淇濆瓨銉囥兗銈夸簰鎻涖儐銈广儓锛堛偔銉笺仺shape锛�

### 4) Module split锛堝焦鍓插垎闆級
- [ ] `halbach/geom.py`锛圧OI鐢熸垚銆佸绉般偆銉炽儑銉冦偗銈癸級
- [ ] `halbach/physics.py`锛圢umba銈兗銉嶃儷锛�
- [ ] `halbach/objective.py`锛堢洰鐨勯枹鏁帮紜瑙ｆ瀽鍕鹃厤锛�
- [ ] `halbach/robust.py`锛圙N姝ｅ墖鍖�/HVP锛�
- [ ] `halbach/io.py`锛坣pz/json/mat淇濆瓨锛�
- [ ] CLI锛堝叆鍙ｏ級銇у緭鏉ャ仺鍚岀瓑銇嫊銇�

### 5) Types & Logging
- [ ] dataclass锛圙eometry, Settings, Results锛�
- [ ] type hints锛坢ypy strict锛�
- [ ] print 銈� logging 銇Щ琛岋紙verbose銇у埗寰★級

### 6) Solver abstraction锛堟嫛寮点伄瓒冲牬锛�
- [ ] Optimizer銈ゃ兂銈裤兗銉曘偋銉笺偣灏庡叆
- [ ] L-BFGS-B瀹熻銈掋偗銉┿偣鍖�
- [ ] 浠婂緦 Gauss-Newton/Diagonal-CCP 銈掕拷鍔犲彲鑳姐伀

### 7) Performance harness锛堥�熷害妞滆ḿ锛�
- [ ] 绨℃槗銉欍兂銉併優銉笺偗銈广偗銉儣銉�
- [ ] 閲嶈銉兗銉椼伄涓嶈銈€儹銈便兗銈枫儳銉冲墛娓�
- [ ] numba parallel 銇瑷庯紙浠绘剰锛�

### 8) Robust term abstraction
- [ ] RobustTerm銈ゃ兂銈裤兗銉曘偋銉笺偣
- [ ] GN瀹熻銈扴trategy鍖栵紙灏嗘潵CVaR/SAA銇拷鍔犱綑鍦帮級

### 9) Documentation锛圕I銇仐鎯冲畾锛�
- [ ] README鏇存柊锛堛偦銉冦儓銈€儍銉�/瀹熻/銉堛儵銉栥儷銈枫儱銉笺儓锛�
- [ ] 銈堛亸銇傘倠銈ㄣ儵銉奸泦锛圵indows/PowerShell/venv/numba锛�

---

## Acceptance criteria锛堟渶绲傚彈銇戝叆銈岋級
- [ ] 銉欍兗銈圭洰鐨勯枹鏁帮紙銉庛儫銉娿儷锛夈仺淇濆瓨銉曘偐銉笺優銉冦儓浜掓彌銇岀董鎸併仌銈屻仸銇勩倠
- [ ] pytest 銇屻儹銉笺偒銉仹閫氥倠
- [ ] mypy strict 銇岋紙numba澧冪晫銈掗櫎銇嶏級閫氥倠
- [ ] 鏈�閬╁寲銉籑C銉讳繚瀛樸亴寰撴潵銇ㄥ悓妲樸伀瀹熻銇с亶銈�

---

## 尰忬偺曽恓乮儊儌乯
- near 廤崌偼嫍棧鑷抣偱愗懼偟側偄丅僀儞僨僢僋僗憢偱屌掕乮埬B乯丅
  - 椺: Wz=1, Wphi=2, Wr=0 傪僨僼僅儖僩岓曗乮彨棃挷惍乯丅
- cube 嬤帡偼摉柺 multi-dipole乮僒僽憃嬌巕廤崌乯偱幚憰偟丄彨棃乽棫曽懱暯嬒僥儞僜儖乿偵抲姺偡傞 TODO 傪曐帩丅
- DC/CCP 偼婛懚 L-BFGS-B 僼儗乕儉偵崿偤偢丄暿僜儖僶僶僢僋僄儞僪偲偟偰捛壛乮徴撍夞旔乯丅
- self-consistent near kernel: multi-dipole (source split only) implemented.
- TODO: replace near kernel with cube-average tensor (analytic/high-accuracy).
- self-consistent JAX objective supports implicit diff via custom_linear_solve (jit cached).
