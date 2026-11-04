import AppKit
import Foundation
import UserNotifications

private struct Settings: Decodable {
    let projectDirectory: String
    let configPath: String
    let pythonPath: String
    /// How long to wait for the disk with the code to appear; five minutes when absent.
    let diskWaitSeconds: Double?

    static func load() -> Settings {
        guard let url = Bundle.main.url(forResource: "settings", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let settings = try? JSONDecoder().decode(Settings.self, from: data) else {
            fatalError("AI-Lab Menu has no settings.json")
        }
        return settings
    }
}

/// What the menu shows: a title, an optional detail line, a colour and a symbol.
private struct Appearance {
    let title: String
    let detail: String
    let color: NSColor?
    let symbol: String
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Where the menu is in starting and keeping AI-Lab up.
    private enum Phase { case stopped, waiting, launched, restarting, error }

    private static let refreshInterval: TimeInterval = 3
    private static let diskCheckInterval: TimeInterval = 5
    private static let restartDelay: TimeInterval = 5
    private static let restartLimit = 3
    private static let restartWindow: TimeInterval = 600
    private static let startupGrace: TimeInterval = 180

    private let settings = Settings.load()
    private var statusItem: NSStatusItem!
    private var statusLine: NSMenuItem!
    private var detailLine: NSMenuItem!
    private var managerProcess: Process?
    private var managerLog: FileHandle?
    private var refreshTimer: Timer?
    private var diskTimer: Timer?

    private var phase = Phase.stopped
    /// True while the owner wants AI-Lab up: set by Start, cleared by Stop and by an error.
    private var wanted = false
    private var reason = ""
    private var waitingSince: Date?
    private var launchedAt: Date?
    private var healthy = false
    private var everHealthy = false
    private var healthInFlight = false
    /// Unexpected exits that were restarted, within the restart window.
    private var restarts: [Date] = []

    private var diskWait: TimeInterval { settings.diskWaitSeconds ?? 300 }

    private var logsDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/AI-Lab")
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        try? FileManager.default.createDirectory(at: logsDirectory, withIntermediateDirectories: true)
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        let menu = NSMenu()
        statusLine = NSMenuItem(title: "AI-Lab: Stopped", action: nil, keyEquivalent: "")
        statusLine.isEnabled = false
        menu.addItem(statusLine)
        detailLine = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        detailLine.isEnabled = false
        detailLine.isHidden = true
        menu.addItem(detailLine)
        menu.addItem(.separator())
        menu.addItem(item("Open AI-Lab", action: #selector(openManager)))
        menu.addItem(.separator())
        menu.addItem(item("Start", action: #selector(startManager)))
        menu.addItem(item("Stop", action: #selector(stopManager)))
        menu.addItem(item("Restart", action: #selector(restartManager)))
        menu.addItem(.separator())
        menu.addItem(item("Show logs", action: #selector(showLogs)))
        statusItem.menu = menu

        refreshStatus()
        startManager()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: Self.refreshInterval, repeats: true) { [weak self] _ in
            self?.checkHealth()
            self?.refreshStatus()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        let stopping = managerProcess
        stopManager()
        stopping?.waitUntilExit()
    }

    private func item(_ title: String, action: Selector) -> NSMenuItem {
        let result = NSMenuItem(title: title, action: action, keyEquivalent: "")
        result.target = self
        return result
    }

    private var managerURL: URL {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: settings.configPath)),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let port = object["port"] as? Int else {
            return URL(string: "http://127.0.0.1:8099")!
        }
        let configuredHost = object["host"] as? String ?? "127.0.0.1"
        let browserHost = configuredHost == "0.0.0.0" ? "127.0.0.1" : configuredHost
        return URL(string: "http://\(browserHost):\(port)")!
    }

    private var running: Bool {
        managerProcess?.isRunning == true
    }

    @objc private func openManager() {
        NSWorkspace.shared.open(managerURL)
    }

    // MARK: - What the menu shows

    private func appearance() -> Appearance {
        let busy = "brain.head.profile.fill"
        switch phase {
        case .stopped:
            return Appearance(title: "AI-Lab: Stopped", detail: "", color: nil, symbol: "brain.head.profile")
        case .waiting:
            let waited = Int(Date().timeIntervalSince(waitingSince ?? Date()))
            return Appearance(
                title: "AI-Lab: Warning — waiting for the disk (\(waited) s of \(Int(diskWait)) s)",
                detail: reason, color: .systemYellow, symbol: busy)
        case .restarting:
            return Appearance(title: "AI-Lab: Warning — restarting", detail: reason,
                              color: .systemYellow, symbol: busy)
        case .error:
            return Appearance(title: "AI-Lab: Error", detail: reason, color: .systemRed, symbol: busy)
        case .launched:
            if healthy {
                return Appearance(title: "AI-Lab: Running", detail: "", color: .systemGreen, symbol: busy)
            }
            let health = managerURL.appendingPathComponent("api/health").absoluteString
            let started = Date().timeIntervalSince(launchedAt ?? Date())
            if !everHealthy && started < Self.startupGrace {
                return Appearance(title: "AI-Lab: Warning — starting", detail: "Waiting for \(health)",
                                  color: .systemYellow, symbol: busy)
            }
            return Appearance(title: "AI-Lab: Warning — not answering", detail: "No answer at \(health)",
                              color: .systemYellow, symbol: busy)
        }
    }

    private func refreshStatus() {
        let look = appearance()
        statusLine.title = look.title
        detailLine.title = look.detail
        detailLine.isHidden = look.detail.isEmpty
        var image = NSImage(systemSymbolName: look.symbol, accessibilityDescription: look.title)
        if let color = look.color, let base = image {
            image = base.withSymbolConfiguration(NSImage.SymbolConfiguration(paletteColors: [color]))
            image?.isTemplate = false
        }
        statusItem.button?.image = image
        statusItem.button?.toolTip = look.detail.isEmpty ? look.title : "\(look.title)\n\(look.detail)"
    }

    /// Asks AI-Lab's open health address whether it answers; one question at a time.
    private func checkHealth() {
        guard phase == .launched, running, !healthInFlight else { return }
        healthInFlight = true
        var request = URLRequest(url: managerURL.appendingPathComponent("api/health"))
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            let answered = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async {
                guard let self else { return }
                self.healthInFlight = false
                guard self.phase == .launched else { return }
                self.healthy = answered
                if answered { self.everHealthy = true }
                self.refreshStatus()
            }
        }.resume()
    }

    // MARK: - Starting and keeping AI-Lab up

    /// The first of the files AI-Lab needs that cannot be read yet, or nil when all are there.
    private func missingPrerequisite() -> String? {
        let files = FileManager.default
        let module = URL(fileURLWithPath: settings.projectDirectory)
            .appendingPathComponent("ai_lab/main.py").path
        if !files.isReadableFile(atPath: module) { return module }
        if !files.isReadableFile(atPath: settings.configPath) { return settings.configPath }
        if !files.isExecutableFile(atPath: settings.pythonPath) { return settings.pythonPath }
        return nil
    }

    @objc private func startManager() {
        wanted = true
        restarts = []
        beginStart()
    }

    /// Starts AI-Lab when its files are there; otherwise waits for them, checking every few seconds.
    private func beginStart() {
        guard wanted, !running else { return }
        if let missing = missingPrerequisite() {
            if phase != .waiting {
                phase = .waiting
                waitingSince = Date()
                record("waiting for \(missing)")
            }
            reason = "Missing: \(missing)"
            if diskTimer == nil {
                diskTimer = Timer.scheduledTimer(withTimeInterval: Self.diskCheckInterval, repeats: true) { [weak self] _ in
                    self?.diskTick()
                }
            }
            refreshStatus()
            return
        }
        stopWaiting()
        launch()
    }

    private func diskTick() {
        guard wanted, phase == .waiting else {
            stopWaiting()
            return
        }
        guard let missing = missingPrerequisite() else {
            record("the files are there")
            beginStart()
            return
        }
        if Date().timeIntervalSince(waitingSince ?? Date()) >= diskWait {
            fail("The disk with the code did not appear in \(Int(diskWait)) s. Missing: \(missing)")
        } else {
            refreshStatus()
        }
    }

    private func stopWaiting() {
        diskTimer?.invalidate()
        diskTimer = nil
        waitingSince = nil
    }

    private func launch() {
        let logURL = logsDirectory.appendingPathComponent("manager.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        managerLog = try? FileHandle(forWritingTo: logURL)
        _ = try? managerLog?.seekToEnd()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: settings.pythonPath)
        process.arguments = ["-m", "ai_lab.main", "--config", settings.configPath]
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
        process.environment = [
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "PYTHONPATH": settings.projectDirectory,
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        ]
        process.standardOutput = managerLog
        process.standardError = managerLog
        process.terminationHandler = { [weak self] finished in
            let how = finished.terminationReason == .uncaughtSignal ? "signal" : "code"
            let ending = "exited with \(how) \(finished.terminationStatus)"
            DispatchQueue.main.async {
                guard let self, self.managerProcess === finished else { return }
                self.managerProcess = nil
                try? self.managerLog?.close()
                self.managerLog = nil
                self.healthy = false
                if self.wanted {
                    self.handleExit(ending)
                } else {
                    self.phase = .stopped
                    self.reason = ""
                    self.record("stopped")
                    self.refreshStatus()
                }
            }
        }

        do {
            try process.run()
            managerProcess = process
            phase = .launched
            launchedAt = Date()
            healthy = false
            everHealthy = false
            reason = ""
            record("started, pid \(process.processIdentifier)")
        } catch {
            try? managerLog?.close()
            managerLog = nil
            managerProcess = nil
            handleExit("could not start: \(error.localizedDescription)")
            return
        }
        refreshStatus()
    }

    /// AI-Lab ended while it was wanted: restart it, unless it already restarted too often.
    private func handleExit(_ ending: String) {
        let now = Date()
        restarts = restarts.filter { now.timeIntervalSince($0) < Self.restartWindow }
        record("AI-Lab \(ending)")
        if restarts.count >= Self.restartLimit {
            fail("AI-Lab \(ending), after \(Self.restartLimit) restarts in "
                 + "\(Int(Self.restartWindow / 60)) minutes. See Show logs.")
            return
        }
        restarts.append(now)
        phase = .restarting
        reason = "AI-Lab \(ending); restart \(restarts.count) of \(Self.restartLimit) in \(Int(Self.restartDelay)) s"
        refreshStatus()
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.restartDelay) { [weak self] in
            guard let self, self.wanted, self.phase == .restarting else { return }
            self.beginStart()
        }
    }

    private func fail(_ why: String) {
        stopWaiting()
        wanted = false
        phase = .error
        reason = why
        record("error: \(why)")
        notify(why)
        refreshStatus()
    }

    @objc private func stopManager() {
        wanted = false
        stopWaiting()
        guard let process = managerProcess, process.isRunning else {
            managerProcess = nil
            phase = .stopped
            reason = ""
            refreshStatus()
            return
        }
        process.terminate()
        refreshStatus()
    }

    @objc private func restartManager() {
        guard let process = managerProcess, process.isRunning else {
            startManager()
            return
        }
        wanted = false
        process.terminate()
        DispatchQueue.global().async { [weak self, weak process] in
            process?.waitUntilExit()
            DispatchQueue.main.async {
                self?.startManager()
            }
        }
    }

    @objc private func showLogs() {
        NSWorkspace.shared.open(logsDirectory)
    }

    // MARK: - Telling the owner

    /// One line in menu-events.log, beside AI-Lab's own log.
    private func record(_ event: String) {
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(event)\n"
        let url = logsDirectory.appendingPathComponent("menu-events.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
            try? handle.close()
        } else {
            try? Data(line.utf8).write(to: url)
        }
    }

    /// A macOS notification; through osascript when the app may not post its own.
    private func notify(_ text: String) {
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { granted in
            let allowed = granted.authorizationStatus == .authorized
                || granted.authorizationStatus == .provisional
            guard allowed else {
                Self.notifyWithScript(text)
                return
            }
            let content = UNMutableNotificationContent()
            content.title = "AI-Lab"
            content.body = text
            content.sound = .default
            center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)) { error in
                if error != nil { Self.notifyWithScript(text) }
            }
        }
    }

    private static func notifyWithScript(_ text: String) {
        let script = Process()
        script.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        script.arguments = [
            "-e", "on run argv",
            "-e", "display notification (item 1 of argv) with title \"AI-Lab\" sound name \"Basso\"",
            "-e", "end run",
            text,
        ]
        try? script.run()
    }
}

let application = NSApplication.shared
let applicationDelegate = AppDelegate()
application.delegate = applicationDelegate
application.run()
