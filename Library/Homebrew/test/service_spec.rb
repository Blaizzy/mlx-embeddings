# typed: false
# frozen_string_literal: true

require "formula"
require "service"

describe Homebrew::Service do
  let(:klass) do
    Class.new(Formula) do
      url "https://brew.sh/test-1.0.tbz"
    end
  end
  let(:name) { "formula_name" }
  let(:path) { Formulary.core_path(name) }
  let(:spec) { :stable }
  let(:f) { klass.new(name, path, spec) }

  describe "#std_service_path_env" do
    it "returns valid std_service_path_env" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
        environment_variables PATH: std_service_path_env
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      path = f.service.std_service_path_env
      expect(path).to eq("#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin")
    end
  end

  describe "#process_type" do
    it "throws for unexpected type" do
      f.class.service do
        run opt_bin/"beanstalkd"
        process_type :cow
      end

      expect {
        f.service.manual_command
      }.to raise_error TypeError, "Service#process_type allows: 'background'/'standard'/'interactive'/'adaptive'"
    end
  end

  describe "#run_type" do
    it "throws for unexpected type" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :cow
      end

      expect {
        f.service.manual_command
      }.to raise_error TypeError, "Service#run_type allows: 'immediate'/'interval'/'cron'"
    end
  end

  describe "#manual_command" do
    it "returns valid manual_command" do
      f.class.service do
        run "#{HOMEBREW_PREFIX}/bin/beanstalkd"
        run_type :immediate
        environment_variables PATH: std_service_path_env, ETC_DIR: etc/"beanstalkd"
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      path = f.service.manual_command
      expect(path).to eq("ETC_DIR=\"#{HOMEBREW_PREFIX}/etc/beanstalkd\" #{HOMEBREW_PREFIX}/bin/beanstalkd")
    end

    it "returns valid manual_command without variables" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
        environment_variables PATH: std_service_path_env
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      path = f.service.manual_command
      expect(path).to eq("#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd")
    end
  end

  describe "#to_plist" do
    it "returns valid plist" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :immediate
        environment_variables PATH: std_service_path_env, FOO: "BAR", ETC_DIR: etc/"beanstalkd"
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        input_path var/"in/beanstalkd"
        root_dir var
        working_dir var
        keep_alive true
        launch_only_once true
        process_type :interactive
        restart_delay 30
        interval 5
        macos_legacy_timers true
      end

      plist = f.service.to_plist
      plist_expect = <<~EOS
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>EnvironmentVariables</key>
        \t<dict>
        \t\t<key>ETC_DIR</key>
        \t\t<string>#{HOMEBREW_PREFIX}/etc/beanstalkd</string>
        \t\t<key>FOO</key>
        \t\t<string>BAR</string>
        \t\t<key>PATH</key>
        \t\t<string>#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        \t</dict>
        \t<key>KeepAlive</key>
        \t<true/>
        \t<key>Label</key>
        \t<string>homebrew.mxcl.formula_name</string>
        \t<key>LaunchOnlyOnce</key>
        \t<true/>
        \t<key>LegacyTimers</key>
        \t<true/>
        \t<key>ProcessType</key>
        \t<string>Interactive</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd</string>
        \t\t<string>test</string>
        \t</array>
        \t<key>RootDirectory</key>
        \t<string>#{HOMEBREW_PREFIX}/var</string>
        \t<key>RunAtLoad</key>
        \t<true/>
        \t<key>StandardErrorPath</key>
        \t<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.error.log</string>
        \t<key>StandardInPath</key>
        \t<string>#{HOMEBREW_PREFIX}/var/in/beanstalkd</string>
        \t<key>StandardOutPath</key>
        \t<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.log</string>
        \t<key>TimeOut</key>
        \t<integer>30</integer>
        \t<key>WorkingDirectory</key>
        \t<string>#{HOMEBREW_PREFIX}/var</string>
        </dict>
        </plist>
      EOS
      expect(plist).to eq(plist_expect)
    end

    it "returns valid partial plist" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
      end

      plist = f.service.to_plist
      plist_expect = <<~EOS
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>Label</key>
        \t<string>homebrew.mxcl.formula_name</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd</string>
        \t</array>
        \t<key>RunAtLoad</key>
        \t<true/>
        </dict>
        </plist>
      EOS
      expect(plist).to eq(plist_expect)
    end

    it "returns valid interval plist" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :interval
        interval 5
      end

      plist = f.service.to_plist
      plist_expect = <<~EOS
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>Label</key>
        \t<string>homebrew.mxcl.formula_name</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd</string>
        \t</array>
        \t<key>RunAtLoad</key>
        \t<false/>
        \t<key>StartInterval</key>
        \t<integer>5</integer>
        </dict>
        </plist>
      EOS
      expect(plist).to eq(plist_expect)
    end

    it "returns valid cron plist" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :cron
        cron "@daily"
      end

      plist = f.service.to_plist
      plist_expect = <<~EOS
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>Label</key>
        \t<string>homebrew.mxcl.formula_name</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd</string>
        \t</array>
        \t<key>RunAtLoad</key>
        \t<false/>
        \t<key>StartCalendarInterval</key>
        \t<dict>
        \t\t<key>Hour</key>
        \t\t<integer>0</integer>
        \t\t<key>Minute</key>
        \t\t<integer>0</integer>
        \t</dict>
        </dict>
        </plist>
      EOS
      expect(plist).to eq(plist_expect)
    end
  end

  describe "#to_systemd_unit" do
    it "returns valid unit" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :immediate
        environment_variables PATH: std_service_path_env, FOO: "BAR"
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        input_path var/"in/beanstalkd"
        root_dir var
        working_dir var
        keep_alive true
        process_type :interactive
        restart_delay 30
        macos_legacy_timers true
      end

      unit = f.service.to_systemd_unit
      std_path = "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin"
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated unit for formula_name

        [Install]
        WantedBy=multi-user.target

        [Service]
        Type=simple
        ExecStart=#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd test
        Restart=always
        RestartSec=30
        WorkingDirectory=#{HOMEBREW_PREFIX}/var
        RootDirectory=#{HOMEBREW_PREFIX}/var
        StandardInput=file:#{HOMEBREW_PREFIX}/var/in/beanstalkd
        StandardOutput=append:#{HOMEBREW_PREFIX}/var/log/beanstalkd.log
        StandardError=append:#{HOMEBREW_PREFIX}/var/log/beanstalkd.error.log
        Environment=\"PATH=#{std_path}\"
        Environment=\"FOO=BAR\"
      EOS
      expect(unit).to eq(unit_expect.strip)
    end

    it "returns valid partial oneshot unit" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
        launch_only_once true
      end

      unit = f.service.to_systemd_unit
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated unit for formula_name

        [Install]
        WantedBy=multi-user.target

        [Service]
        Type=oneshot
        ExecStart=#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd
      EOS
      expect(unit).to eq(unit_expect.strip)
    end
  end

  describe "#to_systemd_timer" do
    it "returns valid timer" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :interval
        interval 5
      end

      unit = f.service.to_systemd_timer
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated timer for formula_name

        [Install]
        WantedBy=timers.target

        [Timer]
        Unit=homebrew.formula_name
        OnUnitActiveSec=5
      EOS
      expect(unit).to eq(unit_expect.strip)
    end

    it "returns valid partial timer" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
      end

      unit = f.service.to_systemd_timer
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated timer for formula_name

        [Install]
        WantedBy=timers.target

        [Timer]
        Unit=homebrew.formula_name
      EOS
      expect(unit).to eq(unit_expect)
    end

    it "throws on incomplete cron" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :cron
        cron "1 2 3 4"
      end

      expect {
        f.service.to_systemd_timer
      }.to raise_error TypeError, "Service#parse_cron expects a valid cron syntax"
    end

    it "returns valid cron timers" do
      styles = {
        "@hourly":   "*-*-*-* *:00:00",
        "@daily":    "*-*-*-* 00:00:00",
        "@weekly":   "0-*-*-* 00:00:00",
        "@monthly":  "*-*-*-1 00:00:00",
        "@yearly":   "*-*-1-1 00:00:00",
        "@annually": "*-*-1-1 00:00:00",
        "5 5 5 5 5": "5-*-5-5 05:05:00",
      }

      styles.each do |cron, calendar|
        f.class.service do
          run opt_bin/"beanstalkd"
          run_type :cron
          cron cron.to_s
        end

        unit = f.service.to_systemd_timer
        unit_expect = <<~EOS
          [Unit]
          Description=Homebrew generated timer for formula_name

          [Install]
          WantedBy=timers.target

          [Timer]
          Unit=homebrew.formula_name
          Persistent=true
          OnCalendar=#{calendar}
        EOS
        expect(unit).to eq(unit_expect.chomp)
      end
    end
  end

  describe "#timed?" do
    it "returns false for immediate" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :immediate
      end

      expect(f.service.timed?).to be(false)
    end

    it "returns true for interval" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :interval
      end

      expect(f.service.timed?).to be(true)
    end
  end

  describe "#command" do
    it "returns @run data" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :immediate
      end

      command = f.service.command
      expect(command).to eq(["#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd", "test"])
    end
  end
end
