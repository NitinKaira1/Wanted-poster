import random
import json

#  TEMPLATE DATA 

CRIMES = [
    "Excessive use of forehead to block WiFi signals",
    "Impersonating someone who has their life together",
    "Crimes against the snooze button (47 counts)",
    "Public disturbance: laughing at own jokes in silence",
    "Hoarding chargers that belong to no known device",
    "Wanted for chronic 'I'll start Monday' fraud",
    "Repeated unauthorized napping during working hours",
    "Aggravated mismatch of socks in broad daylight",
    "Suspicious inability to parallel park with witnesses present",
    "Grand theft: stealing the last slice without asking",
    "Possession of 47 unread emails with intent to ignore",
    "Conspiracy to hit snooze 9 consecutive times",
    "Unlawful possession of 14 browser tabs since 2021",
    "Fraud: pretending to be productive while watching reels",
]

TRAITS = [
    "known to mutter song lyrics incorrectly with full confidence",
    "last seen arguing with a vending machine and losing",
    "believed to survive entirely on instant noodles and spite",
    "considered armed with a truly unbeatable resting face",
    "reportedly allergic to early mornings and responsibility",
    "has a documented history of 'just five more minutes'",
    "operates under the alias 'will reply tomorrow, I promise'",
    "extremely dangerous when within 10 feet of a buffet",
    "known accomplice: a phone battery that dies at 40%",
    "believed to have not replied to a text since 2019",
    "considered highly dangerous near any food labeled 'do not touch'",
]

CLOSING_LINES = [
    "Approach with caffeine. Do not approach before 9 AM.",
    "If spotted, do not make eye contact — it encourages them.",
    "Considered emotionally unstable around free food.",
    "Reward valid only if returned with their dignity intact.",
    "Authorities advise: bribery with snacks may ensure surrender.",
    "Last seen heading toward the fridge at 2 AM.",
    "Do not ask them to make a decision — any decision.",
    "Handle with chai. Two sugars. No questions.",
]

BOUNTIES = [500, 1000, 1337, 4200, 9999, 100, 25000, 69420]

ALIASES = [
    "The Snooze Bandit",
    "Captain No-Reply",
    "The Midnight Snacker",
    "Lord of Unread Emails",
    "The WiFi Blocker",
    "Agent Procrastinator",
    "The Sock Mismatcher",
    "Baron Von Late",
    "The Tab Hoarder",
    "Sir Replies Never",
]

#  TEMPLATE ROAST 

def generate_template_roast():
    """Builds a roast from random template pieces. Works fully offline."""
    crime   = random.choice(CRIMES)
    trait   = random.choice(TRAITS)
    closing = random.choice(CLOSING_LINES)
    bounty  = random.choice(BOUNTIES)
    alias   = random.choice(ALIASES)

    return {
        "alias":       alias,
        "crime":       crime,
        "description": f"{trait.capitalize()}. {closing}",
        "bounty":      bounty,
        "source":      "template",
    }

#  GROQ API ROAST 

def generate_groq_roast(api_key: str):
    """
    Uses Groq's super-fast LLaMA model to generate a unique roast.
    Falls back to template if API call fails.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        prompt = (
            "You are a sarcastic Wild-West sheriff writing a WANTED poster for a "
            "totally harmless but hilariously pathetic criminal. "
            "Return ONLY a valid JSON object — no markdown, no explanation — with exactly these keys:\n"
            "  alias       - a funny 2-4 word nickname\n"
            "  crime       - one absurd, harmless crime (1 sentence)\n"
            "  description - a 1-2 sentence roast of their personality or habits\n"
            "  bounty      - a funny integer dollar amount (no symbols, just the number)\n"
            "Be extremely funny and over-the-top. No real crimes or violence."
        )

        response = client.chat.completions.create(
            model="llama3-70b-8192",   # fast, free, hilarious
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=1.0,           # max creativity
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw.strip())
        data["source"] = "groq"

        # Make sure bounty is an int
        data["bounty"] = int(str(data["bounty"]).replace(",", "").replace("$", ""))

        return data

    except Exception as e:
        print(f"[roast] Groq API failed ({e}), falling back to templates.")
        return generate_template_roast()

#  MAIN ENTRY POINT 

def get_roast(use_api: bool = False, api_key: str = None):
    """Single function the rest of the app calls."""
    if use_api and api_key:
        return generate_groq_roast(api_key)
    return generate_template_roast()


if __name__ == "__main__":
    print("── Template roast ──")
    import pprint
    pprint.pprint(get_roast())
    # To test Groq: pprint.pprint(get_roast(use_api=True, api_key="gsk_..."))
