# IND320 – Project Work Part 1 (Dashboard basics)

> Scaffold جاهز للانطلاق. اتّبع الخطوات في الأسفل لتهيئة المستودع على GitHub ونشر تطبيق Streamlit.

## هيكلة المجلدات
```
ind320_part1_scaffold/
├─ app/                 # كود Streamlit
│  ├─ pages/            # الصفحات 2–4
│  ├─ Home.py           # الصفحة الرئيسية
│  ├─ requirements.txt  # متطلبات النشر
│  └─ (ضع open-meteo-subset.csv هنا)
├─ notebook/
│  └─ part1_dashboard_basics.ipynb
├─ .gitignore
└─ README.md
```

## تشغيل محليًا
1. أنشئ بيئة بايثون (اختياري):  
   - macOS/Linux:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -m venv .venv; .\.venv\Scripts\Activate.ps1
     ```
2. ثبّت المتطلبات:
   ```bash
   pip install -r app/requirements.txt
   ```
3. ضع ملف `open-meteo-subset.csv` داخل مجلد `app/`.
4. شغّل التطبيق:
   ```bash
   streamlit run app/Home.py
   ```

## نشر على Streamlit Cloud
- اربط حسابك على Streamlit بـ GitHub.
- اختر هذا المستودع وحدّد ملف بدء التشغيل: `app/Home.py`.
- تأكد من وجود `app/requirements.txt`.

## ملاحظات
- حدّث مكان الروابط داخل الدفتر (GitHub/Streamlit).
- قبل التصدير إلى PDF: شغّل كل خلايا الدفتر.
