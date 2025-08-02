def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust
import queue
from _pyrepl.commands import end


def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust


def is_chess_960(row):
    indices_rooks = []
    indices_bishops = []
    index_king = []
    index_counter = 0

    for ch in row:
        if ch == "r":
            indices_rooks.append(index_counter)
        if ch == "b":
            indices_bishops.append(index_counter)
        if ch == "K":
            index_king.append(index_counter)
        index_counter += 1
    if (indices_bishops[1] - indices_bishops[0]) % 2 != 0 and indices_rooks[0] < index_king[0] < indices_rooks[1]:
        return True
    else:
        return False

def multiplicative_persistence(n , ignore_zeros = False):
    n_string = str(n)
    persistence = 0
    while len(n_string) > 1:
        product = 1
        for num in n_string:
            if ignore_zeros == True and num =="0":
                continue
            product *= int(num)
        n_string = str(product)
        persistence += 1
    return persistence


def topswops (cards):
    cards = list(cards)
    n = len(cards)
    counter = 0

    while cards != list(range(1, n + 1)):
        k = cards[0]
        if k == 1:
            return counter
        else:
            cards = cards[:k][::-1] + cards[k:]
            counter += 1
    return counter


def is_ascending(items):
    for i in range (len(items)):
        if i == 0:
            continue
        if items[i] <= items[i-1]:
            return False
        else:
            continue
    return True

def riffle(items, out=True):
    result = []
    right_half = items[: 1+len(items)//2]
    left_half = items[len(items)//2 :]
    for i in range(len(left_half)):
        if out == True:
            result.append(right_half[i])
            result.append(left_half[i])
        else:
            result.append(left_half[i])
            result.append(right_half[i])

    return result
def only_odd_digits(n):
    a = str(n)
    for i in range(len(a)):
        if int(a[i]) % 2 == 1:
            continue
        else:
            return False
    return True

def is_cyclops(n):
    a = str(n)
    l = len(a)
    count = 0
    middle_character = a[l // 2]
    if l % 2 == 1 and middle_character == "0":
        for ch in a:
            if ch == "0":
                count += 1
        if count == 1:
            return True
        else:
            return False
    else:
        return False

def domino_cycle(tiles):
    pip_values = [x for tile in tiles for x in tile]
    count = 0
    if len(pip_values)  == 0:
        return True
    pip_values.append(pip_values[0])
    del pip_values[0]

    for i in range(len(pip_values) -1):
        if i%2 == 0:
            if pip_values[i] == pip_values[i+1]:
                count += 1
            else:
                return False
    if count == len(pip_values) // 2:
        return True

def colour_trio(colours):
    def rules(a,b):
        if a == b:
            return a
        if a == 'b':
            if b == 'r':
                return 'y'
            if b == 'y':
                return 'r'
        if a == 'r':
            if b == 'b':
                return 'y'
            if b == 'y':
                return 'b'
        if a == 'y':
            if b == 'r':
                return 'b'
            if b == 'b':
                return 'r'

    if len(colours) == 1:
        return colours
    while len(colours) > 1:
        new_colour = ""

        for i in range(len(colours) - 1):
            new_colour += rules(colours[i], colours[i + 1])
        colours = new_colour

    return colours

def count_dominators(items):
    items = items[::-1]
    count = 1
    maximum = items[0]

    for i in items[1:]:
        if i > maximum:
            count += 1
            maximum = i
    return count

def extract_increasing(digits):
    result = []
    i = 0
    n = len(digits)
    previous = -1

    while i < n:
        current = 0
        j = i
        while j < n:
            current = current * 10 + int(digits[j])
            if current > previous:
                previous = current
                result.append(current)
                break
            j += 1
        i = j + 1
    return result

def lowest_common_dominator(beta, gamma):
    b_sequence, g_sequence, result = [], [], []
    sum_b, sum_g = 0, 0
    for i in range(len(beta)):
        sum_b += beta[i]
        sum_g += gamma[i]
        b_sequence.append(sum_b)
        g_sequence.append(sum_g)
    for i in range(len(gamma)):
        if b_sequence[i] >= g_sequence[i]:
            result.append(b_sequence[i])
        else:
            result.append(g_sequence[i])
    return result

def discrete_rounding(n):
    current = n
    k_list = list(range(2 , n))
    k_list = k_list[::-1]
    for k in k_list:
        i = current // k
        while current % k != 0:
            current = k * (i + 1)
    return current

def tr(text, ch_from, ch_to):
    i, result= 0, ""
    for ch in text:
        if ch in ch_from:
            i = ch_from.index(ch)
            ch = ch_to[i]
            result += ch
        else:
            result += ch
    return result

def count_cigarettes(n, k):
    result = n
    butts = n
    while butts // k > 0:
        additional_cigarettes = butts // k
        result += additional_cigarettes
        butts = additional_cigarettes + butts % k
    return result

def power_prefix(prefix):
    n = 0
    while True:
        condition = True
        N = str(2  ** n)
        if len(N) >= len(prefix):
            for i in range(len(prefix)):
                if prefix[i] != "*" and prefix[i] != N[i]:
                    condition = False
                    break
            if condition:
                result = n
                break

        n += 1
    return result

def lychrel (n, giveup):

    count = 0
    condition = True
    while count < giveup:
        n_string = str(n)
        n_length = len(n_string)

        if n_string == n_string[::-1]:
            return count
        n_inverse = int(n_string[::-1])
        n = int(n_string) + int(n_inverse)
        count += 1
        if str(n) == str(n)[::-1]:
            return count

def word_positions(sentence, word):
    result = []
    word_list = sentence.split()
    for i in range(len(word_list)):
        if word_list[i] == word:
            result.append(i)
    return result

def dfa(rules, text):
    s = 0
    for letter in text:
        s = rules.get((s , letter))
    return s

def powertrain(n):
    n = str(n)
    count = 0
    while len(n) > 1:
        y = 1
        if len(n) % 2 != 0:
            n += "0"
        for i in range(0, len(n), 2):
            x = int(n[i]) ** int(n[i + 1])
            y *= x
        n = str(y)
        count += 1
    return count



def taxi_zum_zum(moves):
    result = [0 , 0]
    direction = 0
    for move in moves:
        if move == "L":
            direction += -1
        if move == "R":
            direction += 1
        if move == "F":
            direction = direction % 4
            if direction == 0:
                result[1] += 1
            if direction == 1:
                result[0] += 1
            if direction == 2:
                result[1] -= 1
            if direction == 3:
                result[0] -= 1
    result = tuple(result)
    return result

def give_change(amount, coins):
    result = []
    for coin in coins:
        while amount >= coin:
            amount -= coin
            result.append(coin)
    return result

def safe_squares_rooks(n, rooks):
    result = []
    rows = []
    columns = []
    open_columns = []
    open_rows = []
    for rook in rooks:
        columns.append(rook[1])
        rows.append(rook[0])
    for i in range(n):
            if i not in columns:
                open_columns.append(i)
    for i in range(n):
            if i not in rows:
                open_rows.append(i)
    for row in open_rows:
        for column in open_columns:
            result.append((row, column))
    answer = len(result)
    return answer

def can_balance(items):
    left_torque = 1
    right_torque = 1
    result = -1
    for i in range(0 , len(items)):
        left_items = items[0 : i ]
        right_items = items[i + 1 :]
        for index, value in enumerate(left_items[::-1], start = 1):
            left_torque += value * index
        for index, value in enumerate(right_items, start = 1):
            right_torque += value * index

        if left_torque == right_torque:
            result = i
            break
        left_torque , right_torque = 0 , 0

    return result

def group_and_skip(n, out, ins):
    result = []
    while n != 0:
        groups = n // out
        result.append(n % out)
        n = ins * groups
    return result

def pyramid_blocks(n, m, h):
    return(h * (1 -3 * h + 2*h*h - 3*m + 3*h*m - 3*n + 3*h*n + 6*m*n)) // 6

def count_growlers(animals):
    count_growlers = 0
    for i in range(len(animals)):
        count_cat = 0
        count_dog = 0
        if animals[i] in ["cat", "dog"]:
            animals_list = animals[:i]
        if animals[i] in ["tac", "god"]:
            animals_list = animals[i + 1:]
        for animal in animals_list:
            if animal in ["cat", "tac"]:
                count_cat += 1
            if animal in ["dog", "god"]:
                count_dog += 1
        if count_dog > count_cat:
            count_growlers += 1
    return count_growlers

def bulgarian_solitaire(piles, k):
    count = 0
    k_list = list(range(1, k + 1))
    while sorted(piles) != k_list:
        add = 0
        for i in range(len(piles)):
            piles[i] -= 1
            add += 1
        piles.append(add)
        piles = [pile for pile in piles if pile != 0]
        count += 1
    return count

def arithmetic_progression(items):
    result = []
    possible_solutions = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            start = items[i]
            length = 1
            stride = items[j] - start
            current = start

            while (current + stride) in items:
                current += stride
                length += 1
            possible_solution = (start, stride, length)
            possible_solutions.append(possible_solution)

    max_length = max(i[2] for i in possible_solutions)
    max_length_tuples = [tuple for tuple in possible_solutions if tuple[2] == max_length]

    min_start = min(i[0] for i in max_length_tuples)
    min_start_tuples = [tuple for tuple in max_length_tuples if tuple[0] == min_start]

    min_stride = min(i[1] for i in min_start_tuples)
    min_stride_tuple = [tuple for tuple in min_start_tuples if tuple[1] == min_stride]

    return min_stride_tuple[0]

def tukeys_ninthers(items):
    while len(items) > 1:
        triplets = [items[i-1: i+2] for i in range(1 , len(items) + 1) if i % 3 == 1]
        for triplet in triplets:
            triplet.sort()
        items = [triplet[1] for triplet in triplets]
    return items[0]

def seven_zero(n):
    def combinations():
        length = 1
        while True:
            for d in range(1, length + 1):
                for j in range(1, d + 1):
                    combination = int("7" * j + "0" * (d - j))
                    yield combination
                length += 1
    if n % 2 != 0 and n % 5 != 0:
        result = ''
        for i in range(1, 100000000):
            result += '7'
            if int(result) % n == 0:
                return int(result)
    else:
        for combination in combinations():
            if combination % n == 0:
                return combination



def parking_lot_permutation(preferred_spot):
    n = len(preferred_spot)
    parking_spots = [None for i in range(n)]
    for i, spot in enumerate(preferred_spot):
        while parking_spots[spot] is not None:
            spot = (spot + 1) % n
        parking_spots[spot] = i
    return parking_spots

def words_with_given_shape(words, shape):
    result = []
    shape_len = len(shape)
    for word in words:
        if len(word) != shape_len + 1:
            continue
        match = True
        for i in range(shape_len):
            diff = 1 if word[i + 1] > word[i] else -1 if word[i + 1] < word[i] else 0
            if diff != shape[i]:
                match = False
                break
        if match:
            result.append(word)
    return result

from itertools import combinations


def count_triangles(sides):
    sides.sort()
    count = 0
    for comb in combinations(sides, 3):
        if comb[0] + comb[1] > comb[2]:
            count += 1
    return count

def arrow_walk(board, position):
    result = 0
    list_board = list(board)
    i = position
    while i in range(len(list_board)):
        if list_board[i] == '<':
            list_board[i] = '>'
            i -= 1
        else:
            list_board[i] = '<'
            i += 1
        result += 1
    return result

import datetime
def count_friday_13s(start, end):
    year = start.year
    month = start.month
    counter = 0
    while year < end.year or (year == end.year and month <= end.month):
        condition = datetime.date(year, month, 13)
        if start <= condition <= end and condition.weekday() == 4:
            counter += 1
        month += 1
        if month == 13:
            month = 1
            year += 1

    return counter

def two_summers(items, goal, i=0, j=None):
    j = len(items) - 1 if j is None else j
    while i < j:
        x = items[i] + items[j]
        if x == goal:
            return True 
        elif x < goal:
            i += 1
        else:
            j -= 1
    return False
def three_summers(items, goal):
    for i in range(len(items)):
        x = goal - items[i]
        list_excluding_item = items[:i] + items[i + 1:]
        if two_summers(list_excluding_item, x):
            return True

    return False


def count_palindromes(text):
    n, count = len(text), 0
    for i in range(1, n-1):
        for j in [0,1]:
            left, right = i, i + j
            while left >= 0 and right < n and text[left] == text[right]:
                if (right - left + 1) >= 3:
                    count += 1
                left -= 1
                right += 1
    return count

def count_carries(a,b):
    count, carry = 0, 0
    while a > 0 or b > 0:
        a_digit = a % 10
        b_digit = b % 10
        digit_sum = a_digit + b_digit + carry
        carry = digit_sum // 10
        if carry != 0:
            count += 1
        a = a // 10
        b = b // 10
    return count

def first_preceded_by_smaller(items, k=1):

    for i in range(k, len(items)):
        count = 0
        check = items[:i]
        for j in check:
            if items[i] > j:
                count += 1
        if count >= k:
            return items[i]
    return None

def reverse_ascending_sublists(items):
    temp_rslt, rslt = [], []

    for i in range(len(items)):
        if not temp_rslt:
            temp_rslt.append(items[i])
        elif items[i] > temp_rslt[-1]:
            temp_rslt.append(items[i])
        else:
            temp_rslt.reverse()
            rslt.extend(temp_rslt)
            temp_rslt = [items[i]]
    if temp_rslt:
        temp_rslt.reverse()
        rslt.extend(temp_rslt)
    return rslt

def collect_numbers(perm):
    n = len(perm)
    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    rounds = 1
    for i in range(1, n):
        if inv_perm[i] < inv_perm[i-1]:
            rounds += 1
    return rounds

def verify_betweenness(perm, constraints):
    inv = [0] * len(perm)
    for i, val in enumerate(perm):
        inv[val] = i
    for a, b, c in constraints:
        if not (min(inv[a], inv[c])) < inv[b] < max(inv[a], inv[c]):
            return False
    return True

def duplicate_digit_bonus(n):
    string_n = str(n)
    score = 0
    k_score = 0
    j = 0
    while j < len(string_n):
        k = 1
        while k + j < len(string_n) and string_n[j] == string_n[j + k]:
            k += 1
        if k > 1:
            if j + k == len(string_n):
                k_score = 2 * (10 ** (k - 2))
                score += k_score
            else:
                k_score = 10 ** (k - 2)
                score += k_score
            j = j + k
        else:
            j += 1
    return score

def expand_intervals(intervals):
    if len(intervals) == 0:
        return[]
    rslt = []
    num_int = intervals.split(",")
    for string in num_int:
        rng = string.split("-")
        if len(rng) == 1:
            rslt.append(int(rng[0]))
        else:
            for i in range(int(rng[0]), int(rng[1])+1):
                rslt.append(i)
    return rslt

def collapse_intervals(items):
    if len(items) < 1:
        return ''
    rslt = str(items[0])
    prev = items[0]
    i = 1
    while i < len(items):
        cntr = 0
        while i < len(items) and items[i] == prev + 1:
            prev = items[i]
            if rslt[-1] != "-":
                rslt += "-"
            cntr += 1
            i += 1
        else:
            if cntr > 0:
                rslt += str(prev)
                rslt += ','
            else:
                if rslt[-1] != ',':
                    rslt += ','
                if i < len(items):
                    rslt += str(items[i])
                    prev = items[i]
                    i += 1
                else:
                    if rslt[-1] == '-':
                        rslt += items[-1]
    if rslt[-1] == ',':
        return rslt[:-1]
    return rslt

def candy_share(candies):
    counter = 0
    while any(candy > 1 for candy in candies):
        candies2 = candies[:]
        for i in range(len(candies)):
            if candies[i] > 1:
                if i == 0:
                    candies2[i] = candies2[i] - 2
                    candies2[i + 1] += 1
                    candies2[-1] += 1
                    continue
                if i == len(candies) - 1:
                    candies2[i] = candies2[i] - 2
                    candies2[0] += 1
                    candies2[i - 1] += 1
                    continue
                else:
                    candies2[i] = candies2[i] - 2
                    candies2[i + 1] += 1
                    candies2[i - 1] += 1
        candies = candies2
        counter += 1
    return counter

def front_back_sort(perm):
    n = len(perm)
    pos = [0] * n
    for i, val in enumerate(perm):
        pos[val] = i
        dp = [1]*n
        for j in range(n-2, -1, -1):
            if pos[j] < pos[j + 1]:
                dp[j] = dp[j + 1] + 1
    return n-max(dp)

def str_rts(text):
    pos, best = dict(), 0
    for (i, c) in enumerate(text):
        if c not in pos:
            pos[c] = [i]
        else:
            pos[c].append(i)
    for c in pos:
        for (i, j) in combinations(pos[c], 2):
            k = 0
            if i + 2*best < j:
                while i< j and text[i] == text[j]:
                    k, i, j = k+1, i+1, j-1
                best = max(k, best)
    return best

























































































































