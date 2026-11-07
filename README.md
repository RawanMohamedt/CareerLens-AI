CareerLens-AI (Updated & Fixed Edition)
نظام متكامل لتحليل السير الذاتية (PDF, DOCX, TXT) ومطابقتها مع توصيف الوظائف (Job Descriptions) وتقديم توصيات تعليمية مخصصة لكل التخصصات (UI/UX Design, Backend, Frontend, Mobile, Data/AI, DevOps, ERP/Odoo).
أهم التحديثات والإصلاحات المدمجة
استخراج الروابط المدمجة في الـ PDF (/Annots): استخراج روابط Behance وLinkedIn والمشاريع المدمجة خلف النصوص في ملفات PDF.
إصلاح استخراج أرقام الهواتف المصرية والدولية: التقاط صيغ 01012345678 و+201112768894 وكافة الصيغ الدولية، ودعم النقطة . في معرفات LinkedIn، وإضافة حقل portfolio.
توسيع قاموس المهارات (skill_extractor.py): دعم مهارات وأدوات UI/UX Design (Figma, Auto Layout, Wireframing, Prototyping, User Research, Design Systems, WCAG, Arabic RTL) بالإضافة لتخصصات البرمجة والبيانات.
تطوير محرك المطابقة (matching_engine.py): تقييم المهارات الفرعية للمتطلبات المركبة، وتنبيه فرق الأقدمية (Seniority Gap)، والتمييز بين الخبرة العملية والمشاريع الطلابية.
تحديث المستشار التعليمي (learning_advisor.py): دعم مصادر التعلم الرسمية للتصميم والبرمجة (Figma Help, Material Design 3, NN/g, W3C, Python, FastAPI, Docker, Odoo).
طريقة التشغيل
تشغيل الملف الشامل الموحد (All-in-One):
streamlit run careerlens_complete.py
أو عبر سطر الأوامر (CLI):
python careerlens_complete.py --cv path/to/resume.pdf --jd path/to/job_description.txt
