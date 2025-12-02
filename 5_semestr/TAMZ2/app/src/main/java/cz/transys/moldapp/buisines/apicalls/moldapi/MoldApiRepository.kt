package cz.transys.moldapp.buisines.apicalls.moldapi

import cz.transys.moldapp.buisines.apicalls.ApiClient
import cz.transys.moldapp.buisines.apicalls.moldapi.CarrierMountResponse
import io.ktor.client.statement.HttpResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext


class MoldApiRepository {

    private var carriersCache: List<CarriersList>? = null
    private var carsCache: List<CarCodeList>? = null

    suspend fun login(empId: String): EmpIdResponse {
        return ApiClient.get("foampad/moldpda/login?empId=$empId")
    }
    suspend fun getAllCars(forceRefresh: Boolean = false): List<CarCodeList> {

        if (!forceRefresh && carsCache != null) {
            return carsCache!!
        }

        val result = ApiClient.get<List<CarCodeList>>("foampad/moldpda/carcode")
        carsCache = result
        return result
    }

    suspend fun getAllCarriers(forceRefresh: Boolean = false): List<CarriersList> {

        if (!forceRefresh && carriersCache != null) {
            return carriersCache!!
        }

        val result = ApiClient.get<List<CarriersList>>("foampad/moldpda/carriers")
        carriersCache = result
        return result
    }

    suspend fun getMoldsByCarCode(carCode: String): List<MoldsList> {
        return ApiClient.get("foampad/moldpda/molds/$carCode")
    }

    suspend fun getCarrierMount(carrier: String): CarrierMountResponse {
        return ApiClient.get("foampad/moldpda/mount/carrier?carrier=$carrier")
    }

    suspend fun checkApiAvailable(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val response: HttpResponse = ApiClient.get("health")
                response.status.value in 200..299
            } catch (e: Exception) {
                false
            }
        }
    }

}

