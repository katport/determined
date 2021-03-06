package resourcemanagers

// calculateDesiredNewAgentNum calculates the new instances based on pending tasks and
// slots per instance.
func calculateDesiredNewAgentNum(
	taskList *taskList, slotsPerAgent int, maxZeroSlotTasksPerAgent int,
) int {
	slotSum := 0
	allTasks := 0
	zeroSlotTasks := 0
	for it := taskList.iterator(); it.next(); {
		// TODO(DET-4035): This code is duplicated from the fitting functions in the
		//    scheduler. To determine is a task is schedulable, we would ideally interface
		//    with the scheduler in some way and not duplicate this logic.
		switch {
		case taskList.GetAllocations(it.value().TaskActor) != nil:
			// If a task is already allocated, skip it.
			continue
		case it.value().SlotsNeeded == 0:
			zeroSlotTasks++
			allTasks++
		case slotsPerAgent == 0:
			continue
		case it.value().SlotsNeeded <= slotsPerAgent:
			slotSum += it.value().SlotsNeeded
			allTasks++
		case it.value().SlotsNeeded%slotsPerAgent == 0:
			slotSum += it.value().SlotsNeeded
			allTasks++
		}
	}

	numAgentByZeroSlot, numAgentBySlot := 0, 0
	switch {
	case zeroSlotTasks == 0:
		numAgentByZeroSlot = 0
	case maxZeroSlotTasksPerAgent == 0:
		numAgentByZeroSlot = 0
	default:
		numAgentByZeroSlot = (zeroSlotTasks + maxZeroSlotTasksPerAgent - 1) / maxZeroSlotTasksPerAgent
	}
	switch {
	case slotSum == 0:
		numAgentBySlot = 0
	case slotsPerAgent == 0:
		numAgentBySlot = 0
	default:
		numAgentBySlot = (slotSum + slotsPerAgent - 1) / slotsPerAgent
	}
	return max(numAgentByZeroSlot, numAgentBySlot)
}
