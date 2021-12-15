def findMaxDivisorBy2(number, minimalSize):
    cptDivisor = 0
    number = float(number)
    while number.is_integer():
        number = number / 2
        cptDivisor += 1
        if number < minimalSize:
            break
    return (cptDivisor - 1), int(number * 2)