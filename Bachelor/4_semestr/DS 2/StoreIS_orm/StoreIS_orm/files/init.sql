
begin transaction;
  insert into Staff values(1, 'Alan', 'Turing', '2006-02-15', null, 'London');
  insert into "User" values(1, 'Antonin', 'Guttman', 'Ostrava', 1);

  insert into Product values(1, 'HP 15-fd0975nc (89A27EA)', 20000);
  insert into Product values(2, 'HP ProBook 450 G9 (723Z9EA#BCM)', 19000);

  insert into Supply values(1, 1, 15000, 100, '2024-01-03');
  insert into Supply values(2, 2, 14000, 200, '2024-01-03');
commit;

select * from "Order";
select * from Staff;
select * from "User";
select * from Product;
select * from Supply;
select * from OrderItem;

begin
  declare @v_id_order int, @v_ret bit;
  set @v_id_order = 2;

  exec InsertProductInOrder @v_id_order out, @p_id_user=1, @p_id_staff=1, 
    @p_id_product=2, @p_quantity=2, @p_ret=@v_ret out;

  print 'v_id_order: ' + cast(@v_id_order as varchar(10));
  print 'v_ret: ' + cast(@v_ret as char(1));
end;