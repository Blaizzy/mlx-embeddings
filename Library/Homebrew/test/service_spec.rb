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

  describe "#to_plist" do
    it "returns valid plist" do
      f.class.service do
        run [opt_bin/"beanstalkd", "test"]
        run_type :immediate
        environment_variables PATH: std_service_path_env
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      plist = f.service.to_plist
      plist_expect = <<~EOS
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>EnvironmentVariables</key>
        \t<dict>
        \t\t<key>PATH</key>
        \t\t<string>#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        \t</dict>
        \t<key>KeepAlive</key>
        \t<true/>
        \t<key>Label</key>
        \t<string>homebrew.mxcl.formula_name</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>#{HOMEBREW_PREFIX}/opt/formula_name/bin/beanstalkd</string>
        \t\t<string>test</string>
        \t</array>
        \t<key>RunAtLoad</key>
        \t<true/>
        \t<key>StandardErrorPath</key>
        \t<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.error.log</string>
        \t<key>StandardOutPath</key>
        \t<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.log</string>
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
        environment_variables PATH: std_service_path_env
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      unit = f.service.to_systemd_unit
      std_path = "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:/usr/sbin:/sbin"
      unit_expect = <<~EOS
        [Unit]
        Description=Homebrew generated unit for formula_name

        [Service]
        Type=simple
        ExecStart=#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd test
        Restart=always
        WorkingDirectory=#{HOMEBREW_PREFIX}/var
        StandardOutput=append:#{HOMEBREW_PREFIX}/var/log/beanstalkd.log
        StandardError=append:#{HOMEBREW_PREFIX}/var/log/beanstalkd.error.log
        Environment=\"PATH=#{std_path}\"
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
