class DescendingStrategy:

    def __init__(self, num_strategies):
        self.num_strategies = num_strategies
        self.remaining_money = 0
        self.auctions_passed = 0
        self.num_auctions = 0

    # name of the strategy - make sure it is unique
    def name(self):
        return "I dont care about price"

    # name of the author of the strategy
    def author(self):
        return "David Pešek"

    # number of auctions that will be simulated - called before the first auction
    def set_num_auctions(self, num_auctions):
        self.num_auctions = num_auctions

    # amount of money available for all auctions - called before the first aution
    def set_money(self, money):
        self.remaining_money = money

    # called after winning an aution with the price that was paid for the object
    def won(self, price):
        self.remaining_money -= price

    # value of the object for this agent - called before every auction
    def set_value(self, value): 
        self.auctions_passed += 1
        self.value = value

    # shows interest in the object for the current price, called in each iteration of each aution
    def interested(self, price, active_strats):
        # descending strategy
        auctions_left = self.num_auctions - self.auctions_passed
        budget_per_auction = self.remaining_money / max(auctions_left, 1)
        threshold = min(self.value, budget_per_auction)
        return price <= threshold
    
class AscendingStrategy:

    def __init__(self, num_strategies):
        self.num_strategies = num_strategies
        self.remaining_money = 0
        self.auctions_passed = 0
        self.num_auctions = 0

    # name of the strategy - make sure it is unique
    def name(self):
        return "I care about price"

    # name of the author of the strategy
    def author(self):
        return "David Pešek"

    # number of auctions that will be simulated - called before the first auction
    def set_num_auctions(self, num_auctions):
        self.num_auctions = num_auctions

    # amount of money available for all auctions - called before the first aution
    def set_money(self, money):
        self.remaining_money = money

    # called after winning an aution with the price that was paid for the object
    def won(self, price):
        self.remaining_money -= price

    # value of the object for this agent - called before every auction
    def set_value(self, value): 
        self.auctions_passed += 1
        self.value = value

    # shows interest in the object for the current price, called in each iteration of each aution
    def interested(self, price, active_strats):
        # ascending strategy
        if (self.auctions_passed/self.num_auctions) < 0.1:
            return price <= self.value*0.3 and price <= self.remaining_money
        if (self.auctions_passed/self.num_auctions) < 0.4:
            return price <= self.value*0.4 and price <= self.remaining_money
        if (self.auctions_passed/self.num_auctions) < 0.8:
            return price <= self.value*0.5 and price <= self.remaining_money
        if (self.auctions_passed/self.num_auctions) < 0.9:
            return price <= self.value*0.8 and price <= self.remaining_money
        return price <= self.value and price <= self.remaining_money
    
def strategy_ascending(num_strategies):
    return AscendingStrategy(num_strategies)

def strategy_descending(num_strategies):
    return DescendingStrategy(num_strategies)