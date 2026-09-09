# راهنمای اجرای فارسی

## اجرای سریع دفترچه

دفترچهٔ `notebooks/Single_Allocation_Hub_Location_Complete.ipynb` مستقل از
بستهٔ `src/` است. در Google Colab آن را باز کنید، در صورت نیاز سلول اختیاری
Google Drive را اجرا کنید، مقدار `DATA_DIR` را به پوشهٔ شامل `data/raw` تغییر
دهید و سپس **Run all** را بزنید. مقدار پیش‌فرض فقط CAB25 را با ۱۰۰ سناریو و
بودجهٔ کوچک CPU اجرا می‌کند. برای اجرای همهٔ داده‌ها،
`RUN_ALL_DATASETS=True` را فقط در صورت نیاز تغییر دهید.

## اجرای بستهٔ پایتون

در لینوکس یا macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[ml,test]'
.venv/bin/python -m pytest
```

در Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e '.[ml,test]'
.venv\Scripts\python -m pytest
```

نمونهٔ ابتکاری CAB25:

```bash
salh-solve --dataset CAB25 --p 3 --alpha 0.5 --beta 0.5 --scenarios 100 --seed 0 --method heuristic
```

آموزش رتبه‌بند مصنوعی و مقایسهٔ زوجی:

```bash
salh-gvns --train-ranker --datasets CAB25 --seeds 0 --p 3 --alpha 0.5 --beta 0.5 --max-iterations 4 --max-evaluations 12 --output-dir outputs/gvns/cab25
```

## تفسیر

مدل تخصیص یگانه است: هر گره دقیقاً به یک هاب باز تخصیص می‌یابد و هر هاب به
خودش تخصیص دارد. برای سناریوهای هم‌احتمال، (K=\lceil\beta S\rceil) هزینهٔ
بدترین K سناریو را تعیین می‌کند. پیاده‌سازی وزن‌دار عمومی نیز
`eta + sum(q_s * max(C_s - eta, 0)) / beta` را برای احتمال‌های صریح نگه
می‌دارد. MILP دودویی مسیر فقط برای نمونهٔ کوچکِ اعتبارسنجی است؛ نتایج
ابتکاری CAB/AP اثبات بهینگی ندارند.

این پروژه یک تطبیق سبک و CPU-ready برای پایان‌نامه است و بازتولید کامل
مقاله‌ها، CBS، DL-CBS، Benders و شبکهٔ آزمایشی بزرگ مقاله را ادعا نمی‌کند.
