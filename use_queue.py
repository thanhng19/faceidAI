import queue

q_frame = queue.Queue(maxsize = 1);

if not q_frame.full():
	q_frame.put(3)
# q_frame.put(2)
# q_frame.pop()
# print (q_frame.get())