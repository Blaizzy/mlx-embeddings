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

  describe "#to_plist" do
    it "returns valid plist" do
      f.class.service do
        run opt_bin/"beanstalkd"
        run_type :immediate
        environment_variables PATH: std_service_path_env
        error_log_path var/"log/beanstalkd.error.log"
        log_path var/"log/beanstalkd.log"
        working_dir var
        keep_alive true
      end

      plist = f.service.to_plist
      expect(plist).to include("<key>Label</key>")
      expect(plist).to include("<string>homebrew.mxcl.#{name}</string>")
      expect(plist).to include("<key>KeepAlive</key>")
      expect(plist).to include("<key>RunAtLoad</key>")
      expect(plist).to include("<key>ProgramArguments</key>")
      expect(plist).to include("<string>#{HOMEBREW_PREFIX}/opt/#{name}/bin/beanstalkd</string>")
      expect(plist).to include("<key>WorkingDirectory</key>")
      expect(plist).to include("<string>#{HOMEBREW_PREFIX}/var</string>")
      expect(plist).to include("<key>StandardOutPath</key>")
      expect(plist).to include("<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.log</string>")
      expect(plist).to include("<key>StandardErrorPath</key>")
      expect(plist).to include("<string>#{HOMEBREW_PREFIX}/var/log/beanstalkd.error.log</string>")
      expect(plist).to include("<key>EnvironmentVariables</key>")
      expect(plist).to include("<key>PATH</key>")
      expect(plist).to include("<string>#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin:")
    end

    it "returns valid partial plist" do
      f.class.service do
        run bin/"beanstalkd"
        run_type :immediate
      end

      plist = f.service.to_plist
      expect(plist).to include("<string>homebrew.mxcl.#{name}</string>")
      expect(plist).to include("<key>Label</key>")
      expect(plist).not_to include("<key>KeepAlive</key>")
      expect(plist).to include("<key>RunAtLoad</key>")
      expect(plist).to include("<key>ProgramArguments</key>")
      expect(plist).not_to include("<key>WorkingDirectory</key>")
      expect(plist).not_to include("<key>StandardOutPath</key>")
      expect(plist).not_to include("<key>StandardErrorPath</key>")
      expect(plist).not_to include("<key>EnvironmentVariables</key>")
    end
  end
end
