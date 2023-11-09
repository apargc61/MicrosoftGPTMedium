
topc = ' swiss'

def first_non_repeating_character(s):
    #create a dictionary to store frequency of characters
    frequency = {}
    for char in s:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    
    # Traverse the string again to find the first non-repeating character
    
    # for char in s:
    #     if frequency[char] == 1:
    #         return char
    # return None
    for index, char in enumerate(s):
        if frequency[char] == 1:
            return index
    return -1
        
    
print(first_non_repeating_character(topc))   
