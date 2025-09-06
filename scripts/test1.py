class ItemIndexFinder:

    def __init__(self, items):
        from collections import defaultdict

        self._items = items
        # itetDict = defaultdict(items)
        itetDict = {i: items[i] for i in range(len(items))}
        self.itemDict = self.itemDict.append(items)
        # self.itemDict = items

    def find_index(self, item):
        # itemList = self.sortList(self._items)

        # return itemList.index(item) if itemList.index(item) else None

        item_index = self.itemDict.index(item)
        for i in range(len(self.itemDict)):
            if self.itemDict[i] == item:
                return i
        return None
    """ O(n) time complexity 
    Dictionary O(1) 
    """

    def sortList(self, lst):
        print(lst)
        list1 = list(sorted(lst))
        print('list1: \n', list1)
        # lst.sort()
        return list1
        # print("Sort List Here")


LIST = [
    "Cherry",
    "Grapes",
    "Grapefruit",
    "Eggplant",
    "Sweet Potato",
    "Lime",
    "Bell Pepper",
    "Pineapple",
    "Tomato",
    "Watermelon",
    "Lemon",
    "Onion",
    "Okra",
    "Spinach",
    "Cucumber",
    "Garlic",
    "Cantaloupe",
    "Broccoli",
    "Asparagus",
    "Cauliflower",
    "Raspberry",
    "Carrot",
    "Potato",
    "Strawberry",
    "Blueberry",
    "Squash",
    "Apple",
    "Apricot",
    "Kiwi",
    "Lettuce",
    "Corn",
    "Plum",
    "Apple",
    "Artichoke",
    "Raspberry",
    "Brussels Sprouts"
]


def main():
    sl = ItemIndexFinder(LIST)

    # Test 1: Apple should be the first element after sorting
    if sl.find_index("Apple") != 0:
        print("❌ Test 1 Failed: Apple should be at index 0")
        return

    # Test 2: Potato should be at index 27 after sorting
    if sl.find_index("Potato") != 27:
        print("❌ Test 2 Failed: Potato should be at index 27")
        return

    # Test 3: The original LIST should not be modified
    if LIST[17] != "Broccoli":
        print(f"❌ Test 3 Failed: LIST[17] was changed, found {LIST[17]}")
        return

    print("✅ All tests passed")


if __name__ == "__main__":
    main()