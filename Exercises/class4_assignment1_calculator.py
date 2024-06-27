'''
Use a chatbot to help code a simple calculator (addition, subtraction, division, multiplication). 
Try adding more features and make it user-friendly with multiple revision prompts. 

Please submit the code file (.py or .ipynb) and the conversation with the AI chatbot (Chat GPT, Gemini, or Claude). 
There should be a way to share the AI conversation via a link, 
but if you can't figure that out, take a few screenshots and submit those with the code.

# The Chat GPT Link of the Coversation
# https://chatgpt.com/share/279208e3-6cfd-4d39-9e4c-89d7ab6de8c6
'''
# # The Final Version of the Code after first Prompt
# def add(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b

# def multiply(a, b):
#     return a * b

# def divide(a, b):
#     if b == 0:
#         return "Error! Division by zero."
#     return a / b

# def exponentiate(a, b):
#     return a ** b

# def modulo(a, b):
#     return a % b

# def calculator():
#     print("Welcome to the Enhanced Calculator")
    
#     memory = None
    
#     while True:
#         print("\nSelect operation:")
#         print("1. Add")
#         print("2. Subtract")
#         print("3. Multiply")
#         print("4. Divide")
#         print("5. Exponentiate")
#         print("6. Modulo")
#         print("7. Clear Memory")
#         print("8. Exit")
        
#         choice = input("Enter choice(1/2/3/4/5/6/7/8): ")

#         if choice in ['1', '2', '3', '4', '5', '6']:
#             try:
#                 if memory is None:
#                     num1 = float(input("Enter first number: "))
#                 else:
#                     use_memory = input(f"Use last result {memory} as the first number? (yes/no): ")
#                     if use_memory.lower() == 'yes':
#                         num1 = memory
#                     else:
#                         num1 = float(input("Enter first number: "))
                
#                 num2 = float(input("Enter second number: "))
#             except ValueError:
#                 print("Invalid input! Please enter numerical values.")
#                 continue

#             if choice == '1':
#                 result = add(num1, num2)
#             elif choice == '2':
#                 result = subtract(num1, num2)
#             elif choice == '3':
#                 result = multiply(num1, num2)
#             elif choice == '4':
#                 result = divide(num1, num2)
#             elif choice == '5':
#                 result = exponentiate(num1, num2)
#             elif choice == '6':
#                 result = modulo(num1, num2)
            
#             print(f"Result: {result}")
#             memory = result
#         elif choice == '7':
#             memory = None
#             print("Memory cleared.")
#         elif choice == '8':
#             print("Exiting the calculator. Goodbye!")
#             break
#         else:
#             print("Invalid choice! Please select a valid operation.")
        
#         next_calculation = input("Do you want to perform another calculation? (yes/no): ")
#         if next_calculation.lower() != 'yes':
#             break

# calculator()


'''
Assuming that the user will follow directions, the calculator works fine.
However, I have noticed some things that some people might look at before completing the finished product.

1. Do you want to perform another calculation? If you type something that isn't yes or no, it will treat it as no. 
Some would add an Error and request to enter yes or no.
2. Use last result as the first number? Has the same problem as number 1.
3. Error for Division of Zero should not be stored in memory. It causes an error if someone uses it for the next calculation.
'''
# I mentioned number one, and the Chatbot fixed number two as well.
# Then, I mentioned number 3, which was fixed.
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error! Division by zero."
    return a / b

def exponentiate(a, b):
    return a ** b

def modulo(a, b):
    return a % b

def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            return user_input
        print(f"Invalid input! Please enter one of the following: {', '.join(valid_options)}")

def calculator():
    print("Welcome to the Enhanced Calculator")
    
    memory = None
    
    while True:
        print("\nSelect operation:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Exponentiate")
        print("6. Modulo")
        print("7. Clear Memory")
        print("8. Exit")
        
        choice = get_valid_input("Enter choice(1/2/3/4/5/6/7/8): ", ['1', '2', '3', '4', '5', '6', '7', '8'])

        if choice in ['1', '2', '3', '4', '5', '6']:
            try:
                if memory is None:
                    num1 = float(input("Enter first number: "))
                else:
                    use_memory = get_valid_input(f"Use last result {memory} as the first number? (yes/no): ", ['yes', 'no'])
                    if use_memory == 'yes':
                        num1 = memory
                    else:
                        num1 = float(input("Enter first number: "))
                
                num2 = float(input("Enter second number: "))
            except ValueError:
                print("Invalid input! Please enter numerical values.")
                continue

            if choice == '1':
                result = add(num1, num2)
            elif choice == '2':
                result = subtract(num1, num2)
            elif choice == '3':
                result = multiply(num1, num2)
            elif choice == '4':
                result = divide(num1, num2)
            elif choice == '5':
                result = exponentiate(num1, num2)
            elif choice == '6':
                result = modulo(num1, num2)
            
            if isinstance(result, str):
                print(result)
            else:
                print(f"Result: {result}")
                memory = result
        elif choice == '7':
            memory = None
            print("Memory cleared.")
        elif choice == '8':
            print("Exiting the calculator. Goodbye!")
            break
        
        next_calculation = get_valid_input("Do you want to perform another calculation? (yes/no): ", ['yes', 'no'])
        if next_calculation == 'no':
            break

calculator()