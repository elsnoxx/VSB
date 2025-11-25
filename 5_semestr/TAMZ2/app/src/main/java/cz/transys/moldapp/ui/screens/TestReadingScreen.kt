package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.R

@Composable
fun TestReadingScreen(onBack: () -> Unit) {
    val scanner = LocalScanner.current
    val scannedItems = remember { mutableStateListOf<String>() }

    val colors = MaterialTheme.colorScheme

    // Scanner listener
    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { scannedData ->
            if (scannedData.isNotBlank()) {
                scannedItems.add(0, scannedData.trim())
            }
        }
    }

    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        // Title
        Text(
            text = stringResource(R.string.test_reading_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Count
        Text(
            text = stringResource(R.string.scanned_count, scannedItems.size),
            fontSize = 16.sp,
            color = colors.onBackground.copy(alpha = 0.7f),
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // List
        Card(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = colors.surface
            )
        ) {
            if (scannedItems.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = stringResource(R.string.no_scans_yet),
                        color = colors.onSurface.copy(alpha = 0.6f)
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(scannedItems) { item ->
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = colors.secondaryContainer
                            )
                        ) {
                            Column(
                                modifier = Modifier.padding(12.dp)
                            ) {
                                Text(
                                    text = item,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 18.sp,
                                    color = colors.onSecondaryContainer
                                )
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(50.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {

            // Clear
            Button(
                onClick = { scannedItems.clear() },
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.tertiary,
                    contentColor = colors.onTertiary
                )
            ) {
                Text(
                    stringResource(R.string.clear_list_button),
                    fontWeight = FontWeight.Bold
                )
            }

            // Back
            Button(
                onClick = onBack,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.secondary,
                    contentColor = colors.onSecondary
                )
            ) {
                Text(
                    stringResource(R.string.back_button),
                    fontWeight = FontWeight.Bold
                )
            }
        }

    }
}

