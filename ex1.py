import datetime
#ex1 by Eyal YAffe
#1.1
print("Hello world")
#1.2
message = "Level Two" 
print(message)
#1.3
print(type(message))
#1.4
a = 123 
b = 654 
c = a + b
#1.6
print(c)
a = 100 
print(a)  # think - should this be 123 or 100? 
c = 50
print(c)  # think - should this be 50 or 777? 
d = 10 + a - c  
print(d)  # think - what should this be now?
#1.7
greeting = 'Hi ' 
name = 'Eyal'
message = greeting + name
print(message) 
#1.8 
age =  31
#print(name + ' is ' + age + ' years old')
#1.9
print(name + ' is ' + str(age) + ' years old')
age = '31' 
print(name + ' is ' + age + ' years old') 
#1.10
bobs_age = 15 
your_age =  31
print(your_age == bobs_age)
#1.11
bob_is_older = bobs_age > your_age 
print(bob_is_older) 
#1.12
money = 500 
phone_cost = 240 
tablet_cost = 260 
total_cost = phone_cost + tablet_cost 
can_afford_both = money >= total_cost 
if can_afford_both:    
    message = "You have enough money for both" 
else:    
    message = "You can't afford both devices" 
print(message)

raspberry_pi = 25
pies = 3 * raspberry_pi
total_cost = total_cost + pies
if total_cost <= money:
    message = "You have enough money for 3 raspberry pies as well" 
else:    
    message = "You can't afford 3 raspberry pies"
print(message)
#1.13
colours = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
print('Black' in colours)
colours.append('Black') 
colours.append('White')
print('Black' in colours)
more_colours = ['Gray', 'Navy', 'Pink'] 
colours.extend(more_colours)  
print(colours)
#1.14
primary_colours = ['Red', 'Blue', 'Yellow'] 
secondary_colours = ['Purple', 'Orange', 'Green']
main_colours = primary_colours + secondary_colours
print(main_colours)
#1.15
print(len(main_colours))
all_colours = colours + main_colours
print(len(all_colours))
#1.16
even_numbers = [2, 4, 6, 8, 10, 12] 
multiples_of_three = [3, 6, 9, 12]
numbers = even_numbers + multiples_of_three
print(numbers, len(numbers))   
numbers_set = set(numbers)
print(numbers_set, len(numbers_set))
colour_set = set(all_colours)
print(colour_set)
#1.17
my_class=['Sarah', 'Bob', 'Jim', 'Tom', 'Lucy', 'Sophie', 'Liz', 'Ed']
for student in my_class:
    print(student)
my_class2=['classmate1','classmate2','etc']
my_class.extend(my_class2)
print(my_class)
for student in my_class:
    print(my_class.index(student)+1,student)
#1.18
full_name = 'Dominic Adrian Smith'
first_letter = full_name[0]
last_letter = full_name[19] 
first_three = full_name[:3] 
last_three = full_name[-3:] 
middle = full_name[8:14]     
print(middle)
#1.19
my_sentence = "Hello, my name is Fred" 
parts = my_sentence.split(',')
print(parts)
print(type(parts))
my_long_sentence = "This is a very very very very very very long sentence" 
parts2 = my_long_sentence.split(' ')
print(parts2,len(parts2))
#1.20
person = ('Bobby', 26)
print(person[0] + ' is ' + str(person[1]) + ' years old') 
students = [('Dave', 12),('Sophia', 13),('Sam', 12), ('Kate', 11),('Daniel', 10)]
for student in students:
    print(student)
#1.21
students=[('eyal', 31, 'subject1'),('david', 25, 'subject2'),('ariel', 45, 'subject3')]
for student in students:
    print(student)
for student in students:
    if student[1]>25:
        print(student)
#1.22
addresses={'Lauren': '0161 5673 890',    
           'Amy': '0115 8901 165',    
           'Daniel': '0114 2290 542',
           'Emergency': '999'
           }
print(addresses['Amy'])
print('David' in addresses)  # [False] # 
print('Daniel' in addresses)  # [True] # 
print('999' in addresses)  # [False] # 
print('999' in addresses.values())  # [True] # 
print(999 in addresses.values())  # [False] 
addresses['Amy'] = '0115 236 359' 
print(addresses['Amy']) 
print('Daniel' in addresses)  # [True] # 
del addresses['Daniel'] # 
print('Daniel' in addresses)  # [False] 
for name in addresses:    
    print(name, addresses[name])
#1.23
sum=0
for x in range(1001):
    numAsString = str(x)
    sumdigits = 0
    for i in range(len(numAsString)):
        sumdigits = sumdigits + int(str(i))
    sum = sum+sumdigits
print(sum)        
#1.24
def max(x,y):
    if x>y:
        return x
    if y>x:
        return y
    else:
        return x
print(max(5,3))
print(max(3,5))
#1.25
#assumes all diffrent 
def max_of_three(x,y,z):
    if x>y and x>z:
        return x
    if y>z and y>x:
        return y
    if z>y and z>y:
        return z
print(max_of_three(1,2,3))
print(max_of_three(3,2,1))
print(max_of_three(1,3,2))
#1.26
def MyLength(string): 
    # Initialize count to zero 
    count=0
    # Counting character in a string 
    for i in string: 
        count+= 1
    # Returning count 
    return count
print(MyLength([1,2,3,4,5])) 
print(MyLength("hello"))
#1.27
def isVowel(char):
    if char=='a' or char=='u' or char=='i' or char=='o' or char=='e':
        return True
    else:
        return False
print(isVowel('a'))    
print(isVowel('b'))
#1.28 
def translate(text):
    ans=""
    for i in range(0,len(text)):
        if not isVowel(text[i]) and text[i]!=' ':
            ans = ans + text[i]+'o'+text[i]
        else:
            ans = ans + text[i]
    return ans        
print(translate('this is fun'))
#part 2
#2.1
def devisibleby7(x):
    ans = x % 7
    if ans==0:
        return True
    else:
        return False

def devisibleby5(x):
    ans = x % 5
    if ans==0:
        return True
    else:
        return False

def devisibleby7andnotby5():
    """:-)"""
    for x in range(2000, 3201):
        if devisibleby7(x) and not devisibleby5(x):
            print(x, end=',')
    print('***')            
devisibleby7andnotby5()

#2.2 TBD
"""def factorial():
    x=input()
    result = 1
    for i in range(2, int(x) + 1):
        result *= i
    return result
print(factorial())"""
"""2.3
y = input()
x = dict()
for i in range(1,int(y)+1):
    x.update({i:i*i})
print(x)"""
#2.4
"""x = input()
y = x.split(',')
print(y)
z = tuple (y)
print(z)"""
#2.5
def square(x):
    return x*x
print(square(5))
#2.6
print(abs.__doc__)
print(int.__doc__)
print(input.__doc__)
print(devisibleby7andnotby5.__doc__)
#2.7.6.a
class Triangle:
    number_of_sides=3
    def __init__(self,ang1,ang2,ang3):
        self.ang1 = ang1
        self.ang2 = ang2
        self.ang3 = ang3
        
    def check_angles(self):
        sum=self.ang1+self.ang2+self.ang3
        return sum==180
    
t1 = Triangle(4,5,6)
t2 = Triangle(90,45,45)
print(t1.number_of_sides)
print(t1.check_angles())
print(t2.check_angles())
my_triangle=Triangle(90,30,60)

class Song:
    def __init__(self,lyrics):
        self.lyrics = lyrics
    def sing_me_a_song(self):
        for i in range(len(self.lyrics)):
            print(self.lyrics[i])

happy_bday = Song(["May god bless you, ","Have a sunshine on you,","Happy Birthday to you!"])
happy_bday.sing_me_a_song() 
    
class Lunch:
    def __init__(self,menu):
        self.menu=menu
    def menu_price(self):
        if self.menu=="menu 1":
            print("Your choice: " + self.menu +"Price 12.00")
        elif self.menu=="menu 2":
            print("Your choice: " + self.menu +"Price 13.40")
        else:
            print("error in menu")
Paul = Lunch("menu 1")
Paul.menu_price()            
     
class Point3D(object):
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    def __repr__(self):
        print("(%d, %d, %d)" % (self.x, self.y, self.z))
my_point = Point3D(1,2,3)
my_point.__repr__()        
#2.8.1
x=dict()
for i in range(101):
    x.update({i:i*i})
print(x)    


def isPrime(number):
    for i in range(2,number):
        if number%i==0:
            return False
    return True  

ans=list()
for i in range(1,101):
    if isPrime(i):
        ans.append(i)
print(ans)
  
print(isPrime(10))
print(isPrime(5))
        
#f=open("eyal.txt","a+") 
#f.write("hello world")
#f.close
#b=open("eyal.txt","w")
#b.close
#first_sentance=b.readline()
#print(first_sentance)

"""f=open("square_roots","w+")
for i in range(1,101):
    x = square(i)
    y=str(x)
    f.write(y)
    f.write("\n")"""

"""with open("eyal.txt","w+") as f:
    data = "input data"
    f.write(data)
with open("eyal.txt","r+") as f:
    data = f.readlines()
    print(data)"""
#2.11.1
data = "eyal is in Deep learning course"
data2 = data.split()
print(data2)    
#2.11.2
"""print(" ".join(data2))
print("please enter first num")
x = input()
print("please enter second num")
y = input()
z = int(x)+int(y)
z=str(z)
print('the sum of %s and %s is %s' % (x,y,z))
print('the sum of {} and {} is {}.'.format(x,y,z))"""

#2.12
x=datetime.datetime.now()
print(x)
now = datetime.datetime.now()
dby =now-datetime.timedelta(days=2)
dif=now+datetime.timedelta(hours=12)
print(dby)    
print(dif)
#2.13
try:
    print(1/0)

except ZeroDivisionError:
    print("You can't divide by zero, you're silly.")
    
    
    
    
    










            