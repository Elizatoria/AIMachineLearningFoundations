'''Code a simple palindrome checker script in python using Chat GPT. Try 
adding a feature with a revisional prompt. (Ex. Make the checker loop, so 
that users can check multiple strings.)'''

'''
# Code a simple palindrome checker script in python

This script includes a function is_palindrome that checks if a given string is a palindrome by:
Removing spaces and converting the string to lowercase to ensure the check is case-insensitive and ignores spaces.
Comparing the string to its reverse (using slicing).
The script also contains an example usage section to demonstrate how the function works with various test strings.
'''
# def is_palindrome(s):
#     # Remove spaces and convert to lowercase for uniformity
#     s = s.replace(" ", "").lower()
#     # Check if the string is equal to its reverse
#     return s == s[::-1]

# # Example usage
# if __name__ == "__main__":
#     test_strings = ["racecar", "palindrome", "A man a plan a canal Panama", "No lemon, no melon"]
#     for string in test_strings:
#         result = "is" if is_palindrome(string) else "is not"
#         print(f'"{string}" {result} a palindrome.')


'''
# Make the checker loop

is_palindrome(s) function: This function removes spaces and converts the string to lowercase to ensure the check is case-insensitive 
and ignores spaces. It then checks if the string is equal to its reverse.
Main Loop:
The loop prompts the user to enter a string.
If the user types "exit", the loop breaks and the script ends.
Otherwise, it checks if the entered string is a palindrome and prints the result.
'''
# def is_palindrome(s):
#     # Remove spaces and convert to lowercase for uniformity
#     s = s.replace(" ", "").lower()
#     # Check if the string is equal to its reverse
#     return s == s[::-1]

# if __name__ == "__main__":
#     while True:
#         user_input = input("Enter a string to check (or type 'exit' to quit): ")
#         if user_input.lower() == 'exit':
#             print("Exiting the palindrome checker.")
#             break
#         if is_palindrome(user_input):
#             print(f'"{user_input}" is a palindrome.')
#         else:
#             print(f'"{user_input}" is not a palindrome.')


'''
# Used the whole original text

Here's a simple Python script to check if a string is a palindrome, 
along with a feature to allow users to check multiple strings in a loop:

How it works:
is_palindrome function: This function takes a string, removes any non-alphanumeric characters, 
converts it to lowercase, and checks if it reads the same forwards and backwards.
palindrome_checker function: This function runs a loop, asking the user to enter a string to check if it's a palindrome. 
If the user types 'exit', the loop breaks and the program ends.
Running the Script:
Copy and paste this script into a Python environment or save it as a .py file and run it. 
The script will prompt you to enter strings to check if they are palindromes and will continue until you type 'exit'.
'''
def is_palindrome(s):
    # Remove non-alphanumeric characters and convert to lowercase
    clean_s = ''.join(filter(str.isalnum, s)).lower()
    # Check if the cleaned string is equal to its reverse
    return clean_s == clean_s[::-1]

def palindrome_checker():
    print("Welcome to the Palindrome Checker!")
    while True:
        user_input = input("Enter a string to check (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if is_palindrome(user_input):
            print(f"'{user_input}' is a palindrome!")
        else:
            print(f"'{user_input}' is not a palindrome.")

# Run the palindrome checker
palindrome_checker()