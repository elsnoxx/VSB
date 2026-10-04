create or alter procedure InsertProductInOrder(@p_id_order int out, @p_id_user int, @p_id_staff int, 
  @p_id_product int, @p_quantity int, @p_ret bit out)
as
begin
  set @p_ret = 0;

  begin transaction;

  begin try
    if @p_id_order is null
	begin
      insert into "Order" (id_user, id_staff, date_order, price)
      values(@p_id_user, @p_id_staff, getdate(), null);
	  set @p_id_order = @@identity;
    end

    declare @v_quantity int;
	select @v_quantity = ( 
	  coalesce((select sum(quantity) from Supply where id_product=@p_id_product), 0) -
	  coalesce((select sum(quantity) from OrderItem where id_product=@p_id_product), 0)
	);

    if @v_quantity >= @p_quantity
	begin
      insert into OrderItem(id_order, id_product, unit_price, quantity) 
	  values (@p_id_order, @p_id_product,
        (select p.unit_price *
          (select 
             case u.level
               when 2 then 0.95
               when 3 then 0.9
               when 4 then 0.85
               when 5 then 0.8
               when 6 then 0.75
               when 7 then 0.7
               else 1
             end
           from "User" u where id_user = @p_id_user
          )
        from Product p
        where p.id_product = @p_id_product
      ), @p_quantity);

	  set @p_ret = 1;
	  commit;
	end;
	else rollback;
  end try
  begin catch
    rollback;
  end catch
end;