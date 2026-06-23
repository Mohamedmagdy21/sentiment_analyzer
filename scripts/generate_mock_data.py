#!/usr/bin/env python3
"""Generate a mock customer dataset for pipeline validation.

Creates a CSV with 1,300 records containing customer_id, customer_name,
phone_number, and text columns. The text field contains a mix of positive,
neutral, and negative customer reviews suitable for sentiment classification.

Usage:
    python scripts/generate_mock_data.py [--output mock_customers.csv] [--count 1300]
"""
import argparse
import csv
import random
import os

POSITIVE_TEXTS = [
    "Absolutely love this product, exceeded all my expectations!",
    "The customer service team was incredibly helpful and responsive.",
    "Best purchase I have made all year, highly recommend to everyone.",
    "Fantastic quality and fast shipping, will definitely buy again.",
    "This completely solved my problem, I am so grateful for this product.",
    "Amazing experience from start to finish, five stars all the way.",
    "The team went above and beyond to make sure I was satisfied.",
    "I have been a loyal customer for years and they never disappoint.",
    "The new feature is a game changer, makes my life so much easier.",
    "Wonderful packaging and the product works flawlessly, very impressed.",
    "Their return policy is hassle-free and the staff was very kind.",
    "I recommended this to all my friends and they love it too.",
    "The app is so intuitive and well-designed, great user experience.",
    "Shipping was lightning fast and everything arrived in perfect condition.",
    "This is exactly what I needed, the quality is outstanding.",
]

NEGATIVE_TEXTS = [
    "Terrible experience, the product broke after just two days of use.",
    "Customer service was rude and unhelpful, would not recommend.",
    "Complete waste of money, the quality is absolutely atrocious.",
    "My order arrived damaged and nobody is responding to my emails.",
    "The worst customer service I have ever experienced in my entire life.",
    "I have been waiting three weeks for my refund, this is unacceptable.",
    "The product does not work as advertised, very misleading description.",
    "Every time I call support I get put on hold for an hour, ridiculous.",
    "The app keeps crashing and nobody has fixed it for months.",
    "I am extremely disappointed with the quality, definitely returning this.",
    "They charged me twice and refused to refund the duplicate charge.",
    "The delivery was late by two weeks and nobody apologized or explained.",
    "Poor build quality, feels cheap and flimsy, not worth the price at all.",
    "I tried to cancel my order but there is no way to do it online.",
    "Absolutely frustrated with the entire process, will never shop here again.",
]

NEUTRAL_TEXTS = [
    "The product arrived on time and functions as described.",
    "Average experience overall, nothing particularly good or bad to report.",
    "It works fine for basic use but nothing special.",
    "The packaging was standard and delivery was on the expected date.",
    "I have been using it for a week, seems okay so far.",
    "The interface is straightforward, does what it says on the tin.",
    "Not bad, not great, just an ordinary product for the price.",
    "The customer service responded within a day, issue was partially resolved.",
    "Quality is acceptable for the price point, no major complaints.",
    "It does the job but I expected slightly better for what I paid.",
    "Standard product, standard service, nothing stood out either way.",
    "The features are basic but functional, meets minimum expectations.",
    "I received the correct order, it works, that is about it.",
    "Neither impressed nor disappointed, it is just a product.",
    "The experience was unremarkable, I have no strong opinion either way.",
]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
    "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret",
    "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly", "Paul",
    "Emily", "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Carol",
    "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa", "Edward",
    "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
]


def generate_phone():
    """Generate a random US-format phone number."""
    return f"({random.randint(200, 989)}) {random.randint(200, 989)}-{random.randint(1000, 9999)}"


def generate_customer_id(i):
    """Generate a customer ID with a consistent prefix."""
    return f"CUST-{i:05d}"


def main():
    parser = argparse.ArgumentParser(description="Generate mock customer data")
    parser.add_argument("--output", default="mock_customers_1300.csv",
                        help="Output CSV filename")
    parser.add_argument("--count", type=int, default=1300,
                        help="Number of records to generate")
    args = parser.parse_args()

    random.seed(42)

    rows = []
    for i in range(1, args.count + 1):
        sentiment_choice = random.choices(
            ["positive", "negative", "neutral"],
            weights=[40, 35, 25],
            k=1
        )[0]

        if sentiment_choice == "positive":
            text = random.choice(POSITIVE_TEXTS)
        elif sentiment_choice == "negative":
            text = random.choice(NEGATIVE_TEXTS)
        else:
            text = random.choice(NEUTRAL_TEXTS)

        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)

        rows.append({
            "customer_id": generate_customer_id(i),
            "customer_name": f"{first} {last}",
            "phone_number": generate_phone(),
            "text": text,
        })

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["customer_id", "customer_name", "phone_number", "text"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {args.count} mock customer records at: {output_path}")

    pos_count = sum(1 for r in rows if r["text"] in POSITIVE_TEXTS)
    neg_count = sum(1 for r in rows if r["text"] in NEGATIVE_TEXTS)
    neu_count = sum(1 for r in rows if r["text"] in NEUTRAL_TEXTS)
    print(f"  Expected distribution: ~{pos_count} positive, ~{neg_count} negative, ~{neu_count} neutral")
    print(f"  CRM action list expected: top 10% of {neg_count} negatives = ~{max(1, int(neg_count * 0.10))} records")


if __name__ == "__main__":
    main()
