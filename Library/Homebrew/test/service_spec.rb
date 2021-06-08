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
        restart_delay 30
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
        \t<key>LegacyTimers</key>
        \t<true/>
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

    it "returns valid partial unit" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
      end

      unit = f.service.to_systemd_unit
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated unit for formula_name

        [Install]
        WantedBy=multi-user.target

        [Service]
        Type=simple
        ExecStart=#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd
      EOS
      expect(unit).to eq(unit_expect)
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
