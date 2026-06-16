# -*- coding: utf-8 -*-
import re, sys, logging
logging.basicConfig(level=logging.WARNING)

print("\n" + "="*50)
print("   AGH Nutrition AI")
print("="*50)
print("\n⏳ Loading models...\n")

try:
    from nutrition_llm import init, UserProfile, CoachProfile, ChatMessage, analyze_meal, chat_coach, plan_meals
    init(model_path="qwen25_onnx", yolo_model="models/best_ep12.onnx")
    print("✅ Models ready!\n")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── Translations ──────────────────────────────────────────────────────────────
UI = {
"Français": dict(
    diet_prompt="🥗 Type de régime :", goal_prompt="🎯 Objectif [Perte de poids]: ",
    goal_default="Perte de poids", stats_title="📊 Statistiques (Entrée = défaut):",
    weekly="   Progrès hebdo [-2.3kg]: ", cal_target="   Objectif calorique [1450]: ",
    cal_today="   Calories aujourd'hui [980]: ", water="   Eau en L [1.2]: ",
    ready="✅ Coach prêt ! ('quitter' pour sortir)", you="Vous", coach="Coach",
    img_path="📁 Chemin image: ", no_path="❌ Aucun chemin.", detecting="⏳ Détection...\n",
    detected="✅ Détecté", no_food="⚠️  Aucun aliment détecté.",
    add_weights="⚖️  Ajouter poids? (o/n) [n]: ", weight_item="   {item} (défaut 100g): ",
    analysis_title="🤖 Analyse IA:\n", prefs_prompt="❤️  Préférences [aucune]: ",
    allergies_prompt="⚠️  Allergies [Aucune]: ", calories_prompt="🔥 Objectif calorique [1600]: ",
    plan_title="🤖 Génération du plan...\n", experience_prompt="📈 Expérience [Débutant]: ",
    experience_default="Débutant", quit_words=["quitter","quit","q"], yes_words=["o","oui","y","yes"],
    diets={"1":"Équilibré","2":"Keto","3":"Végétalien","4":"Méditerranéen"},
    menu=["1. 💬 Chat avec BH Coach","2. 📷 Analyser un repas","3. 🍽️  Plan repas","4. 🚪 Quitter"],
    choice_prompt="\nVotre choix : ",
),
"English": dict(
    diet_prompt="🥗 Diet type:", goal_prompt="🎯 Goal [Weight loss]: ",
    goal_default="Weight loss", stats_title="📊 Your stats (Enter = default):",
    weekly="   Weekly progress [-2.3kg]: ", cal_target="   Calorie target [1450]: ",
    cal_today="   Calories today [980]: ", water="   Water in L [1.2]: ",
    ready="✅ Coach ready! ('quit' to exit)", you="You", coach="Coach",
    img_path="📁 Image path: ", no_path="❌ No path provided.", detecting="⏳ Detecting...\n",
    detected="✅ Detected", no_food="⚠️  No food detected.",
    add_weights="⚖️  Add weights? (y/n) [n]: ", weight_item="   {item} (default 100g): ",
    analysis_title="🤖 AI Analysis:\n", prefs_prompt="❤️  Preferences [none]: ",
    allergies_prompt="⚠️  Allergies [None]: ", calories_prompt="🔥 Calorie target [1600]: ",
    plan_title="🤖 Generating plan...\n", experience_prompt="📈 Experience [Beginner]: ",
    experience_default="Beginner", quit_words=["quit","exit","q"], yes_words=["y","yes"],
    diets={"1":"Balanced","2":"Keto","3":"Vegan","4":"Mediterranean"},
    menu=["1. 💬 Chat with BH Coach","2. 📷 Analyze a meal","3. 🍽️  Meal plan","4. 🚪 Exit"],
    choice_prompt="\nYour choice: ",
),
"Arabic": dict(
    diet_prompt="🥗 نوع النظام الغذائي:", goal_prompt="🎯 الهدف [فقدان الوزن]: ",
    goal_default="فقدان الوزن", stats_title="📊 إحصائياتك (Enter = افتراضي):",
    weekly="   التقدم الأسبوعي [-2.3kg]: ", cal_target="   هدف السعرات [1450]: ",
    cal_today="   السعرات اليوم [980]: ", water="   الماء باللتر [1.2]: ",
    ready="✅ الكوتش جاهز! ('خروج' للخروج)", you="أنت", coach="الكوتش",
    img_path="📁 مسار الصورة: ", no_path="❌ لم يتم توفير مسار.", detecting="⏳ جاري التحليل...\n",
    detected="✅ تم اكتشاف", no_food="⚠️  لم يتم اكتشاف طعام.",
    add_weights="⚖️  إضافة الأوزان؟ (نعم/لا) [لا]: ", weight_item="   {item} (افتراضي 100g): ",
    analysis_title="🤖 تحليل الذكاء الاصطناعي:\n", prefs_prompt="❤️  تفضيلاتك [لا شيء]: ",
    allergies_prompt="⚠️  الحساسية [لا شيء]: ", calories_prompt="🔥 هدف السعرات [1600]: ",
    plan_title="🤖 جاري إنشاء خطة الوجبات...\n", experience_prompt="📈 المستوى [مبتدئ]: ",
    experience_default="مبتدئ", quit_words=["خروج","quit","q"], yes_words=["نعم","y","yes"],
    diets={"1":"متوازن","2":"كيتو","3":"نباتي","4":"متوسطي"},
    menu=["1. 💬 محادثة مع الكوتش","2. 📷 تحليل صورة وجبة","3. 🍽️  خطة وجبات","4. 🚪 خروج"],
    choice_prompt="\nاختيارك: ",
),

"Português": dict(
    diet_prompt="🥗 Tipo de dieta:", goal_prompt="🎯 Objetivo [Perda de peso]: ",
    goal_default="Perda de peso", stats_title="📊 Suas estatísticas (Enter = padrão):",
    weekly="   Progresso semanal [-2.3kg]: ", cal_target="   Meta calórica [1450]: ",
    cal_today="   Calorias hoje [980]: ", water="   Água em L [1.2]: ",
    ready="✅ Coach pronto! ('sair' para sair)", you="Você", coach="Coach",
    img_path="📁 Caminho da imagem: ", no_path="❌ Nenhum caminho fornecido.", detecting="⏳ Detectando...\n",
    detected="✅ Detectado", no_food="⚠️  Nenhum alimento detectado.",
    add_weights="⚖️  Adicionar pesos? (s/n) [n]: ", weight_item="   {item} (padrão 100g): ",
    analysis_title="🤖 Análise IA:\n", prefs_prompt="❤️  Preferências [nenhuma]: ",
    allergies_prompt="⚠️  Alergias [Nenhuma]: ", calories_prompt="🔥 Meta calórica [1600]: ",
    plan_title="🤖 Gerando plano de refeições...\n", experience_prompt="📈 Experiência [Iniciante]: ",
    experience_default="Iniciante", quit_words=["sair","quit","q"], yes_words=["s","sim","y","yes"],
    diets={"1":"Equilibrado","2":"Keto","3":"Vegano","4":"Mediterrâneo"},
    menu=["1. 💬 Chat com BH Coach","2. 📷 Analisar refeição","3. 🍽️  Plano de refeições","4. 🚪 Sair"],
    choice_prompt="\nSua escolha: ",
),
"Italiano": dict(
    diet_prompt="🥗 Tipo di dieta:", goal_prompt="🎯 Obiettivo [Perdita di peso]: ",
    goal_default="Perdita di peso", stats_title="📊 Le tue statistiche (Invio = predefinito):",
    weekly="   Progresso settimanale [-2.3kg]: ", cal_target="   Obiettivo calorico [1450]: ",
    cal_today="   Calorie oggi [980]: ", water="   Acqua in L [1.2]: ",
    ready="✅ Coach pronto! ('esci' per uscire)", you="Tu", coach="Coach",
    img_path="📁 Percorso immagine: ", no_path="❌ Nessun percorso fornito.", detecting="⏳ Rilevamento...\n",
    detected="✅ Rilevato", no_food="⚠️  Nessun alimento rilevato.",
    add_weights="⚖️  Aggiungere pesi? (s/n) [n]: ", weight_item="   {item} (predefinito 100g): ",
    analysis_title="🤖 Analisi IA:\n", prefs_prompt="❤️  Preferenze [nessuna]: ",
    allergies_prompt="⚠️  Allergie [Nessuna]: ", calories_prompt="🔥 Obiettivo calorico [1600]: ",
    plan_title="🤖 Generazione piano pasti...\n", experience_prompt="📈 Esperienza [Principiante]: ",
    experience_default="Principiante", quit_words=["esci","quit","q"], yes_words=["s","si","y","yes"],
    diets={"1":"Equilibrato","2":"Keto","3":"Vegano","4":"Mediterraneo"},
    menu=["1. 💬 Chat con BH Coach","2. 📷 Analizza pasto","3. 🍽️  Piano pasti","4. 🚪 Esci"],
    choice_prompt="\nLa tua scelta: ",
),
"Español": dict(
    diet_prompt="🥗 Tipo de dieta:", goal_prompt="🎯 Objetivo [Pérdida de peso]: ",
    goal_default="Pérdida de peso", stats_title="📊 Tus estadísticas (Enter = predeterminado):",
    weekly="   Progreso semanal [-2.3kg]: ", cal_target="   Meta calórica [1450]: ",
    cal_today="   Calorías hoy [980]: ", water="   Agua en L [1.2]: ",
    ready="✅ Coach listo! ('salir' para salir)", you="Tú", coach="Coach",
    img_path="📁 Ruta de imagen: ", no_path="❌ No se proporcionó ruta.", detecting="⏳ Detectando...\n",
    detected="✅ Detectado", no_food="⚠️  No se detectaron alimentos.",
    add_weights="⚖️  Agregar pesos? (s/n) [n]: ", weight_item="   {item} (predeterminado 100g): ",
    analysis_title="🤖 Análisis IA:\n", prefs_prompt="❤️  Preferencias [ninguna]: ",
    allergies_prompt="⚠️  Alergias [Ninguna]: ", calories_prompt="🔥 Meta calórica [1600]: ",
    plan_title="🤖 Generando plan de comidas...\n", experience_prompt="📈 Experiencia [Principiante]: ",
    experience_default="Principiante", quit_words=["salir","quit","q"], yes_words=["s","si","y","yes"],
    diets={"1":"Equilibrado","2":"Keto","3":"Vegano","4":"Mediterráneo"},
    menu=["1. 💬 Chat con BH Coach","2. 📷 Analizar comida","3. 🍽️  Plan de comidas","4. 🚪 Salir"],
    choice_prompt="\nTu elección: ",
),
"Deutsch": dict(
    diet_prompt="🥗 Diättyp:", goal_prompt="🎯 Ziel [Gewichtsverlust]: ",
    goal_default="Gewichtsverlust", stats_title="📊 Deine Statistiken (Enter = Standard):",
    weekly="   Wöchentlicher Fortschritt [-2.3kg]: ", cal_target="   Kalorienziel [1450]: ",
    cal_today="   Kalorien heute [980]: ", water="   Wasser in L [1.2]: ",
    ready="✅ Coach bereit! ('beenden' zum Beenden)", you="Du", coach="Coach",
    img_path="📁 Bildpfad: ", no_path="❌ Kein Pfad angegeben.", detecting="⏳ Erkennung...\n",
    detected="✅ Erkannt", no_food="⚠️  Keine Lebensmittel erkannt.",
    add_weights="⚖️  Gewichte hinzufügen? (j/n) [n]: ", weight_item="   {item} (Standard 100g): ",
    analysis_title="🤖 KI-Analyse:\n", prefs_prompt="❤️  Präferenzen [keine]: ",
    allergies_prompt="⚠️  Allergien [Keine]: ", calories_prompt="🔥 Kalorienziel [1600]: ",
    plan_title="🤖 Ernährungsplan wird erstellt...\n", experience_prompt="📈 Erfahrung [Anfänger]: ",
    experience_default="Anfänger", quit_words=["beenden","quit","q"], yes_words=["j","ja","y","yes"],
    diets={"1":"Ausgewogen","2":"Keto","3":"Vegan","4":"Mediterran"},
    menu=["1. 💬 Chat mit BH Coach","2. 📷 Mahlzeit analysieren","3. 🍽️  Ernährungsplan","4. 🚪 Beenden"],
    choice_prompt="\nDeine Wahl: ",
),
"中文": dict(
    diet_prompt="🥗 饮食类型：", goal_prompt="🎯 目标 [减肥]：",
    goal_default="减肥", stats_title="📊 您的统计数据（Enter = 默认）：",
    weekly="   每周进度 [-2.3kg]：", cal_target="   卡路里目标 [1450]：",
    cal_today="   今日卡路里 [980]：", water="   今日饮水量（升）[1.2]：",
    ready="✅ 教练准备好了！（输入'退出'退出）", you="您", coach="教练",
    img_path="📁 图片路径：", no_path="❌ 未提供路径。", detecting="⏳ 检测中...\n",
    detected="✅ 已检测到", no_food="⚠️  未检测到食物。",
    add_weights="⚖️  添加重量？（是/否）[否]：", weight_item="   {item}（默认100克）：",
    analysis_title="🤖 AI分析：\n", prefs_prompt="❤️  饮食偏好 [无]：",
    allergies_prompt="⚠️  过敏 [无]：", calories_prompt="🔥 卡路里目标 [1600]：",
    plan_title="🤖 正在生成饮食计划...\n", experience_prompt="📈 经验 [初学者]：",
    experience_default="初学者", quit_words=["退出","quit","q"], yes_words=["是","y","yes"],
    diets={"1":"均衡","2":"生酮","3":"素食","4":"地中海"},
    menu=["1. 💬 与BH教练聊天","2. 📷 分析餐食照片","3. 🍽️  生成饮食计划","4. 🚪 退出"],
    choice_prompt="\n您的选择：",
),
}

TAG_COLORS = {"Motivation":"\033[92m","Conseil":"\033[93m","Question":"\033[91m","RESET":"\033[0m"}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def divider(): print("\n" + "─"*50 + "\n")

def pick_language():
    print("🌐 Language / اللغة / Langue / 语言:")
    options = [
        ("1","Français"),("2","English"),("3","العربية (Arabic)"),
        ("4","Português"),("5","Italiano"),("6","Español"),("7","Deutsch"),("8","中文 (Chinese)"),
    ]
    mapping = {
        "1":"Français","2":"English","3":"Arabic",
        "4":"Português","5":"Italiano","6":"Español","7":"Deutsch","8":"中文",
    }
    for k,v in options: print(f"   {k}. {v}")
    c = input("   Choice [1]: ").strip() or "1"
    lang = mapping.get(c,"Français")
    return lang, UI[lang]

def pick_diet(t):
    print(t["diet_prompt"])
    for k,v in t["diets"].items(): print(f"   {k}. {v}")
    c = input("   [1]: ").strip() or "1"
    return t["diets"].get(c, list(t["diets"].values())[0])

def stream_print(gen, show_tag=False):
    full = ""
    for chunk in gen:
        print(chunk, end="", flush=True)
        full += chunk
    print()
    if show_tag:
        m = re.match(r'^\[(Motivation|Conseil|Question)\]', full.strip())
        if m:
            tag = m.group(1)
            print(f"\n  → {TAG_COLORS.get(tag,'')}{tag}{TAG_COLORS['RESET']}")
    return full

# ─── Coach Chat ────────────────────────────────────────────────────────────────
def run_coach(lang, t):
    divider()
    print("💬 BH COACH\n")
    diet = pick_diet(t)
    goal = input(t["goal_prompt"]).strip() or t["goal_default"]
    print("\n" + t["stats_title"])
    w  = input(t["weekly"]).strip()     or "-2.3kg"
    ct = input(t["cal_target"]).strip() or "1450"
    cd = input(t["cal_today"]).strip()  or "980"
    wa = input(t["water"]).strip()      or "1.2"
    profile = CoachProfile(language=lang, diet_type=diet, goal=goal,
                           weekly_progress=w, calorie_target=int(ct),
                           calories_today=int(cd), water_today=float(wa))
    user_id = f"user_{lang}"   # in production, use real user ID from auth
    print(f"\n{t['ready']}\n")
    divider()
    while True:
        msg = input(f"{t['you']}: ").strip()
        if not msg: continue
        if msg.lower() in t["quit_words"]: break
        print(f"\n{t['coach']}: ", end="", flush=True)
        # Memory handles history automatically — just pass current message
        stream_print(chat_coach([ChatMessage(role="user", content=msg)], profile, user_id=user_id), show_tag=True)
        print()

# ─── Meal Analysis ─────────────────────────────────────────────────────────────
def run_meal_analysis(lang, t):
    divider()
    print("📷 MEAL ANALYSIS\n")
    path = input(t["img_path"]).strip()
    if not path: print(t["no_path"]); return
    diet = pick_diet(t)
    goal = input(t["goal_prompt"]).strip() or t["goal_default"]
    exp  = input(t["experience_prompt"]).strip() or t["experience_default"]
    profile = UserProfile(language=lang, diet_type=diet, goal=goal, experience=exp)
    print("\n" + t["detecting"])
    try:
        detection, stream = analyze_meal(path, profile)
    except FileNotFoundError:
        print(f"❌ {path}"); return
    except Exception as e:
        print(f"❌ {e}"); return
    if not detection.food_items: print(t["no_food"]); return
    print(f"{t['detected']}: {', '.join(detection.food_items)}\n")
    weights = {}
    if input(t["add_weights"]).strip().lower() in t["yes_words"]:
        for item in detection.food_items:
            wg = input(t["weight_item"].format(item=item)).strip() or "100"
            weights[item] = int(wg)
        detection, stream = analyze_meal(path, profile, weights=weights)
    divider()
    print(t["analysis_title"])
    stream_print(stream)
    divider()

# ─── Meal Planner ──────────────────────────────────────────────────────────────
def run_meal_planner(lang, t):
    divider()
    print("🍽️  MEAL PLANNER\n")
    diet      = pick_diet(t)
    goal      = input(t["goal_prompt"]).strip()      or t["goal_default"]
    calories  = input(t["calories_prompt"]).strip()  or "1600"
    prefs     = input(t["prefs_prompt"]).strip()     or ""
    allergies = input(t["allergies_prompt"]).strip() or "None"
    profile   = UserProfile(language=lang, diet_type=diet, goal=goal)
    divider()
    print(t["plan_title"])
    stream_print(plan_meals(profile, int(calories), prefs, allergies))
    divider()

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    lang, t = pick_language()
    while True:
        print("\n" + "="*50 + "\n   AGH Nutrition AI\n" + "="*50)
        for line in t["menu"]: print(" ", line)
        print("="*50)
        c = input(t["choice_prompt"]).strip()
        if c == "1":   run_coach(lang, t)
        elif c == "2": run_meal_analysis(lang, t)
        elif c == "3": run_meal_planner(lang, t)
        elif c in ("4","q","quit","exit","خروج","退出","beenden","sair","salir","esci"): print("\n👋\n"); break
        else: print("❌ 1 / 2 / 3 / 4")

if __name__ == "__main__":
    main()
