print("Total, Top, Rounds")
totals = range(10, 1000, 1)
tops = range(2, 50, 1)

for total in totals:
    for top in tops:
        curr_count = 0

        rounds = 0
        while curr_count < total:
            rounds += 1
            count = curr_count / total
            count = min(count * 2, 1)
            count *= top
            count = max(int(count), 1)
            curr_count += count
        print(f"{total}, {top}, {rounds}")
