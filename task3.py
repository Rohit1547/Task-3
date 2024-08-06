#TASK 3
import re
def assess_password_strength(password):
    strength_criteria = {
        "length": len(password) >= 8,
        "uppercase": re.search(r"[A-Z]", password) is not None,
        "lowercase": re.search(r"[a-z]", password) is not None,
        "digits": re.search(r"\d", password) is not None,
        "special_characters": re.search(r"[!@#$%^&*(),.?\":{}|<>]", password) is not None
    }

    strength_score = sum(strength_criteria.values())

    if strength_score == 5:
        strength = "Very Strong"
    elif strength_score == 4:
        strength = "Strong"
    elif strength_score == 3:
        strength = "Moderate"
    elif strength_score == 2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    feedback = "Password Strength: {}\n".format(strength)
    feedback += "Criteria met:\n"
    for criteria, met in strength_criteria.items():
        feedback += "- {}: {}\n".format(criteria.capitalize(), "Yes" if met else "No")

    return feedback

def main():
    while True:
        password = input("Enter a password to assess its strength: ")
        feedback = assess_password_strength(password)
        print(feedback)

        another = input("Do you want to assess another password? (Y/N): ").upper()
        if another != 'Y':
            break

if __name__ == "__main__":
    main()
