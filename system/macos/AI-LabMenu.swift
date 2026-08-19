import AppKit
import Foundation

private struct Settings: Decodable {
    let projectDirectory: String
    let configPath: String
    let pythonPath: String

    static func load() -> Settings {
        guard let url = Bundle.main.url(forResource: "settings", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let settings = try? JSONDecoder().decode(Settings.self, from: data) else {
            fatalError("AI-Lab Menu has no settings.json")
        }
        return settings
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let settings = Settings.load()
    private var statusItem: NSStatusItem!
    private var statusLine: NSMenuItem!
    private var managerProcess: Process?
    private var managerLog: FileHandle?
    private var refreshTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.toolTip = "AI-Lab"

        let menu = NSMenu()
        statusLine = NSMenuItem(title: "Manager: stopped", action: nil, keyEquivalent: "")
        statusLine.isEnabled = false
        menu.addItem(statusLine)
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
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            self?.refreshStatus()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        stopManager()
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

    private func refreshStatus() {
        statusLine.title = running ? "Manager: running" : "Manager: stopped"
        statusItem.button?.image = NSImage(
            systemSymbolName: running ? "brain.head.profile.fill" : "brain.head.profile",
            accessibilityDescription: running ? "AI-Lab running" : "AI-Lab stopped"
        )
    }

    @objc private func openManager() {
        NSWorkspace.shared.open(managerURL)
    }

    @objc private func startManager() {
        guard !running else { return }

        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/AI-Lab")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let logURL = logs.appendingPathComponent("manager.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        managerLog = try? FileHandle(forWritingTo: logURL)
        _ = try? managerLog?.seekToEnd()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: settings.pythonPath)
        process.arguments = ["-m", "ai_lab.main", "--config", settings.configPath]
        process.currentDirectoryURL = URL(fileURLWithPath: settings.projectDirectory)
        process.environment = [
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        ]
        process.standardOutput = managerLog
        process.standardError = managerLog
        process.terminationHandler = { [weak self, weak process] _ in
            DispatchQueue.main.async {
                guard let self, self.managerProcess === process else { return }
                self.managerProcess = nil
                try? self.managerLog?.close()
                self.managerLog = nil
                self.refreshStatus()
            }
        }

        do {
            try process.run()
            managerProcess = process
        } catch {
            try? managerLog?.close()
            managerLog = nil
            managerProcess = nil
        }
        refreshStatus()
    }

    @objc private func stopManager() {
        guard let process = managerProcess, process.isRunning else {
            managerProcess = nil
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
        process.terminate()
        DispatchQueue.global().async { [weak self, weak process] in
            process?.waitUntilExit()
            DispatchQueue.main.async {
                guard let self else { return }
                self.startManager()
            }
        }
    }

    @objc private func showLogs() {
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/AI-Lab")
        NSWorkspace.shared.open(logs)
    }
}

let application = NSApplication.shared
let applicationDelegate = AppDelegate()
application.delegate = applicationDelegate
application.run()
