intersections = list();
			for y in range(1,len(skeleton)-1):
				for x in range(1,len(skeleton[y])-1):
					if skeleton[y][x] == 1:
						neighbourCount = 0;
						#neighbours = neighbourCoords(x,y);
						for n in neighbours:
							if (skeleton[n[1]][n[0]] == 1):
								neighbourCount += 1;
						if(neighbourCount > 2):
							print(neighbourCount,x,y);
							intersections.append((x,y));
			