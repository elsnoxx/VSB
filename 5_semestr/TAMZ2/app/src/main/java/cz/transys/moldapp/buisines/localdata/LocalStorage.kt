package cz.transys.moldapp.buisines.localdata

import android.content.Context

class LocalStorage(context: Context) {
    private val prefs = context.getSharedPreferences("moldapp_prefs", Context.MODE_PRIVATE)

    fun saveUserId(userId: String) {
        prefs.edit().putString("user_id", userId).apply()
    }

    fun getUserId(): String? {
        return prefs.getString("user_id", null)
    }

    fun clearUserId() {
        prefs.edit().remove("user_id").apply()
    }
}