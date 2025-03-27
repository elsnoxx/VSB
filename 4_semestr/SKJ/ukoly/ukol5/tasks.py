import xml.etree.ElementTree as ET

def create_student(xml_root, student_id):
    '''
    Vytvořte studenta dle loginu.
    Ujistěte se, že student neexistuje, jinak: raise Exception('student already exists')
    '''
    if xml_root.find(f"student[@student_id='{student_id}']") is not None:
        raise Exception('student already exists')
    student = ET.Element('student')
    student.set('student_id', student_id)
    xml_root.append(student)
    return xml_root
    pass


def remove_student(xml_root, student_id):
    '''
    Odstraňte studenta dle loginu
    '''
    if xml_root.find(f"student[@student_id='{student_id}']") is not None:
        xml_root.remove(xml_root.find(f"student[@student_id='{student_id}']"))
    else:
        raise Exception('student not found')
    pass


def set_task_points(xml_root, student_id, task_id, points):
    '''
    Přepište body danému studentovi u jednoho tasku
    '''
    if xml_root.find(f"student[@student_id='{student_id}']/task[@task_id='{task_id}']") is not None:
        xml_root.find(f"student[@student_id='{student_id}']/task[@task_id='{task_id}']").text = str(points)
    else:
        raise Exception('task not found')
    pass


def create_task(xml_root, student_id, task_id, points):
    '''
    Pro daného studenta vytvořte task s body.
    Ujistěte se, že task (s task_id) u studenta neexistuje, jinak: raise Exception('task already exists')
    '''

    pass


def remove_task(xml_root, task_id):
    '''
    Napříč všemi studenty smažte task s daným task_id
    '''

    pass
