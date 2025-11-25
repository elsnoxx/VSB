package cz.transys.moldapp.ui.apicalls


class MoldApiRepository {

    suspend fun getAllCars(): List<CarCodeList> {
        return ApiClient.get("foampad/moldpda/carcode")
    }

    suspend fun getAllCarriers(): List<CarriersList> {
        return ApiClient.get("foampad/moldpda/carriers")
    }

    suspend fun getMoldsByCarCode(carCode: String): List<MoldsList> {
        return ApiClient.get("foampad/moldpda/molds/$carCode")
    }
}


// Příklad datového modelu
data class CarCodeList(
    val car_code: String,
    val car_name: String
)

data class CarriersList(
    val code_value1: String,
    val code_value2: String
)

data class MoldsList(
    val mold_code: String,
    val mold_side: String,
    val code_value2: String
)