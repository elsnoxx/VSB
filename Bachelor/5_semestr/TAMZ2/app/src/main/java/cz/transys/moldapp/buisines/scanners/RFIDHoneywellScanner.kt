package cz.transys.moldapp.buisines.scanners

import android.content.Context
import com.honeywell.rfidservice.EventListener
import com.honeywell.rfidservice.RfidManager
import com.honeywell.rfidservice.rfid.Gen2
import com.honeywell.rfidservice.rfid.OnTagReadListener
import com.honeywell.rfidservice.rfid.RfidReader
import com.honeywell.rfidservice.rfid.TagAdditionData
import com.honeywell.rfidservice.rfid.TagReadData
import com.honeywell.rfidservice.rfid.TagReadOption
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class RFIDHoneywellScanner(private val context: Context) {

    private val _tags = MutableStateFlow<List<String>>(emptyList())
    val tags: StateFlow<List<String>> = _tags

    private var mRfidManager: RfidManager? = null
    private var mRfidReader: RfidReader? = null

    fun open() {
        RfidManager.create(context, object : RfidManager.CreatedCallback {
            override fun onCreated(rfidManager: RfidManager) {
                mRfidManager = rfidManager
                rfidManager.addEventListener(object : EventListener() {
                    override fun onDeviceConnected(data: Any?) {
                        rfidManager.createReader()
                    }

                    override fun onDeviceDisconnected(data: Any?) {}

                    override fun onReaderCreated(success: Boolean, reader: RfidReader?) {
                        if (success && reader != null) {
                            try {
                                reader.setWorkMode(Gen2.Session.Session1, 4, 0, -1)
                                reader.setRegion(com.honeywell.rfidservice.rfid.Region.NA)
                            } catch (e: Exception) {
                                // ignore region/workmode errors for now
                            }
                            reader.setOnTagReadListener(object : OnTagReadListener {
                                override fun onTagRead(t: Array<TagReadData>) {
                                    val list = t.map { tag ->
                                        val add = tag.getAdditionData()
                                        tag.getEpcHexStr() + ":" + bytes2HexStr(add, 0, add.size, "", false)
                                    }
                                    _tags.value = (_tags.value + list).distinct()
                                }
                            })
                            mRfidReader = reader
                        }
                    }
                })
            }
        })
    }

    fun connect(mac: String) = mRfidManager?.connect(mac)
    fun startRead() {
        val opt = TagReadOption().apply { setData(true) }
        mRfidReader?.read(TagAdditionData.TID_BANK, opt)
    }
    fun stopRead() = mRfidReader?.stopRead()
    fun close() {
        try {
            mRfidReader?.stopRead()
        } catch (_: Exception) {}
        mRfidReader = null
        mRfidManager = null
    }

    companion object {
        private val HEX_ARRAY = "0123456789ABCDEF".toCharArray()

        private fun bytes2HexStr(
            bytes: ByteArray?, offset: Int, lengthInput: Int,
            joinStrInput: String?, reverse: Boolean
        ): String {
            if (bytes == null) return ""
            var length = lengthInput
            val offsetVar = offset
            var joinStr = joinStrInput ?: ""
            if (length <= 0 || offsetVar >= bytes.size) return ""
            if (length > bytes.size - offsetVar) length = bytes.size - offsetVar
            val hexChs = CharArray((length shl 1) + joinStr.length * (length - 1))
            val joinChs = joinStr.toCharArray()
            val end = offsetVar + length
            if (reverse) {
                var m = 0
                for (i in end - 1 downTo offsetVar) {
                    if (i != end - 1) {
                        for (jc in joinChs) hexChs[m++] = jc
                    }
                    val v = bytes[i].toInt() and 0xFF
                    hexChs[m++] = HEX_ARRAY[v ushr 4]
                    hexChs[m++] = HEX_ARRAY[v and 0x0F]
                }
            } else {
                var m = 0
                for (i in offsetVar until end) {
                    if (i != offsetVar) {
                        for (jc in joinChs) hexChs[m++] = jc
                    }
                    val v = bytes[i].toInt() and 0xFF
                    hexChs[m++] = HEX_ARRAY[v ushr 4]
                    hexChs[m++] = HEX_ARRAY[v and 0x0F]
                }
            }
            return String(hexChs)
        }
    }
}