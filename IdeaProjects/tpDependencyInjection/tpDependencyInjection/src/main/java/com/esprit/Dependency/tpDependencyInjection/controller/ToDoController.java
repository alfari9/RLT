package com.esprit.Dependency.tpDependencyInjection.controller;

import java.util.List;

import com.esprit.Dependency.tpDependencyInjection.dao.IToDoDao;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Controller;

@Controller
@AllArgsConstructor
public class ToDoController {

	private final IToDoDao toDoService;

	public List<String> getCoursesList() {
		return toDoService.getCoursesList();
	}
}

