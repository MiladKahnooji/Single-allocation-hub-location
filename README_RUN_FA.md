<div dir="rtl" style="text-align: right; font-family: Tahoma, Arial, sans-serif; line-height: 1.9;">

# راهنمای اجرای تحویل

این بسته با <span dir="ltr">Python 3.12</span> اجرا می‌شود. ابتدا در ریشهٔ پوشهٔ تحویل، در لینوکس یا macOS فرمان زیر را اجرا کنید:

<pre dir="ltr" style="text-align:left">chmod +x run_setup.sh
./run_setup.sh</pre>

در ویندوز، محیط مجازی را با <span dir="ltr">py -3.12 -m venv .venv</span> بسازید، سپس با <span dir="ltr">.venv\Scripts\python -m pip install -r requirements.txt</span> وابستگی‌ها را نصب و با <span dir="ltr">.venv\Scripts\python -m pip install -e .</span> پروژه را نصب کنید. آزمون‌ها با <span dir="ltr">.venv\Scripts\python -m pytest</span> اجرا می‌شوند.

## ترتیب دفترچه‌ها

ابتدا <span dir="ltr">notebooks/Single_Allocation_Hub_Location_Complete.ipynb</span> را باز و همهٔ سلول‌ها را از بالا به پایین اجرا کنید. پیکربندی قابل تغییر در ابتدای دفترچه است. مسیر داده‌ها به‌صورت پیش‌فرض <span dir="ltr">data/raw</span> است. سپس برای تولید نمودارهای مقاله‌گونه، <span dir="ltr">notebooks/article_presentation_figures.ipynb</span> را اجرا کنید.

هر دو دفترچه از ۱۰۰ سناریوی وزن‌دار استفاده می‌کنند. برای بازتولید، <span dir="ltr">scenario seed</span> و <span dir="ltr">search seed</span> را ثابت نگه دارید. خروجی‌های تولیدشده زیر <span dir="ltr">outputs/</span> قرار می‌گیرند و جزو کد منبع نیستند.

## تفسیر روش‌ها

<span dir="ltr">Exact MILP</span> فقط برای اعتبارسنجی کوچک است. <span dir="ltr">Benders</span> در اجرای محدود با <span dir="ltr">max_iterations</span> و در صورت نیاز <span dir="ltr">time_limit</span> جواب شدنی، کران پایین <span dir="ltr">LB</span> و کران بالا <span dir="ltr">UB</span> را گزارش می‌کند. شکاف مطلق برابر <span dir="ltr">UB − LB</span> و شکاف نسبی برابر <span dir="ltr">(UB − LB) / max(|UB|, 1e−12)</span> است. فقط وقتی حل‌گر همگرایی را اثبات کند، نتیجهٔ <span dir="ltr">Benders</span> بهینه نامیده می‌شود.

<span dir="ltr">GVNS</span>، <span dir="ltr">CBS/RCBS</span>، <span dir="ltr">DL-GVNS</span> و <span dir="ltr">DL-CBS/DL-RCBS</span> روش‌های ابتکاری‌اند و هرگز بهینهٔ اثبات‌شده نیستند. نسخه‌های <span dir="ltr">DL</span> از رتبه‌های مدل <span dir="ltr">DLHr</span> که فقط روی دادهٔ مصنوعی آموزش دیده است استفاده می‌کنند؛ داده‌های CAB/AP برای آموزش برچسب به کار نمی‌روند.

</div>
