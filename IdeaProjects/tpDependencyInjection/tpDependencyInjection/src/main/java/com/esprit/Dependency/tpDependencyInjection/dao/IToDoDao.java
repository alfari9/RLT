package com.esprit.Dependency.tpDependencyInjection.dao;

import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface IToDoDao {
	public List<String> getCoursesList();
}
