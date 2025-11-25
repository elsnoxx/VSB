package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.R
import cz.transys.moldapp.ui.apicalls.MoldRfInfoRepository
import cz.transys.moldapp.ui.apicalls.TagInfo
import kotlinx.coroutines.launch

@Composable
fun RfTagInfoScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    val repo = remember { MoldRfInfoRepository() }
    val scope = rememberCoroutineScope()

    var rfTag by remember { mutableStateOf("") }
    var info by remember { mutableStateOf<TagInfo?>(null) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    fun loadData(tag: String) {
        scope.launch {
            loading = true
            error = null
            info = null
            try {
                val result = repo.getTagInfo(tag)
                if (result == null) {
                    error = context.getString(R.string.error_tag_not_found)
                } else {
                    info = result
                }
            } catch (e: Exception) {
                error = context.getString(R.string.error_api, e.localizedMessage ?: "")
            } finally {
                loading = false
            }
        }
    }

    fun onScanned(value: String) {
        val clean = value.trim()
        rfTag = clean
        if (clean.isNotEmpty()) loadData(clean)
    }

    val scanner = LocalScanner.current
    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { onScanned(it) }
    }
    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title
        Text(
            text = stringResource(R.string.rf_tag_info_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // Container card
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface, MaterialTheme.shapes.medium)
                .padding(16.dp)
        ) {
            // RF-TAG input
            OutlinedTextField(
                value = rfTag,
                readOnly = true,
                onValueChange = { rfTag = it },
                label = { Text(stringResource(R.string.rf_tag_label)) },
                placeholder = { Text(stringResource(R.string.rf_tag_placeholder)) },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = colors.primary,
                    unfocusedBorderColor = colors.outline,
                    focusedLabelColor = colors.primary,
                    cursorColor = colors.primary
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            // API result
            if (info != null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(colors.surfaceVariant, MaterialTheme.shapes.medium)
                        .padding(12.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        ReadOnlyField(stringResource(R.string.field_mold_id), info!!.mold)
                        ReadOnlyField(stringResource(R.string.field_part_type), info!!.type)
                        ReadOnlyField(stringResource(R.string.field_car), info!!.car)
                        ReadOnlyField(stringResource(R.string.field_status), info!!.status)
                        ReadOnlyField(stringResource(R.string.field_total_cycles), info!!.total)
                    }
                }
            }

            if (error != null) {
                Text(
                    text = error!!,
                    color = colors.error,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // LOAD
                Button(
                    onClick = { loadData(rfTag) },
                    enabled = rfTag.isNotBlank() && !loading,
                    modifier = Modifier
                        .weight(1f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.tertiary,
                        contentColor = colors.onTertiary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.load_data_button),
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }

                // BACK
                Button(
                    onClick = onBack,
                    modifier = Modifier
                        .weight(.5f)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colors.secondary,
                        contentColor = colors.onSecondary
                    )
                ) {
                    Text(
                        text = stringResource(R.string.back_button),
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp
                    )
                }
            }
        }
    }
}

@Composable
fun ReadOnlyField(label: String, value: String) {
    val colors = MaterialTheme.colorScheme

    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = colors.onSurfaceVariant
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface, MaterialTheme.shapes.small)
                .padding(10.dp)
        ) {
            Text(
                text = value.ifBlank { "-" },
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.SemiBold,
                color = colors.onSurface
            )
        }
    }
}

