# typed: true
# frozen_string_literal: true

require "context"
require "erb"
require "settings"
require "api"

module Utils
  # Helper module for fetching and reporting analytics data.
  #
  # @api private
  module Analytics
    INFLUX_BUCKET = "analytics"
    INFLUX_TOKEN = "cSVYFCexszYqAy2h7PpURZuud3VuZzVXtajCtvoPbACseY_XD9LqqGHdKg5W9QAyxYWoHT6jnJHFJzlsy5X2Rg=="
    INFLUX_HOST = "https://eu-central-1-1.aws.cloud2.influxdata.com"
    INFLUX_ORG = "d81a3e6d582d485f"

    class << self
      extend T::Sig

      include Context

      sig { params(type: Symbol, metadata: T::Hash[Symbol, T.untyped]).void }
      def report_google(type, metadata = {})
        analytics_ids = ENV.fetch("HOMEBREW_ANALYTICS_IDS", "").split(",")
        analytics_ids.each do |analytics_id|
          args = []

          # do not load .curlrc unless requested (must be the first argument)
          args << "--disable" unless Homebrew::EnvConfig.curlrc?

          args += %W[
            --max-time 3
            --user-agent #{HOMEBREW_USER_AGENT_CURL}
            --data v=1
            --data aip=1
            --data t=#{type}
            --data tid=#{analytics_id}
            --data uid=n0thxg00gl3
            --data an=#{HOMEBREW_PRODUCT}
            --data av=#{HOMEBREW_VERSION}
          ]
          metadata.each do |key, value|
            next unless value

            key = ERB::Util.url_encode key
            value = ERB::Util.url_encode value
            args << "--data" << "#{key}=#{value}"
          end

          curl = Utils::Curl.curl_executable

          # Send analytics. Don't send or store any personally identifiable information.
          # https://docs.brew.sh/Analytics
          # https://developers.google.com/analytics/devguides/collection/protocol/v1/devguide
          # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters
          if ENV["HOMEBREW_ANALYTICS_DEBUG"]
            url = "https://www.google-analytics.com/debug/collect"
            puts "#{curl} #{args.join(" ")} #{url}"
            puts Utils.popen_read(curl, *args, url)
          else
            pid = fork do
              exec curl, *args,
                   "--silent", "--output", "/dev/null",
                   "https://www.google-analytics.com/collect"
            end
            Process.detach T.must(pid)
          end
        end
        nil
      end

      sig {
        params(measurement: Symbol, package_name: String, tap_name: String, on_request: T::Boolean,
               options: String).void
      }
      def report_influx(measurement, package_name:, tap_name:, on_request:, options:)
        # ensure on_request is a boolean
        on_request = on_request ? true : false

        # ensure options are removed (by `.compact` below) if empty
        options = nil if options.blank?

        # Tags are always implicitly strings and must have low cardinality.
        tags = default_tags_influx.merge(on_request: on_request)
                                  .map { |k, v| "#{k}=#{v}" }
                                  .join(",")

        # Fields need explicitly wrapped with quotes and can have high cardinality.
        fields = default_fields_influx.merge(package: package_name, tap_name: tap_name, options: options)
                                      .compact
                                      .map { |k, v| %Q(#{k}="#{v}") }
                                      .join(",")

        args = [
          "--max-time", "3",
          "--header", "Authorization: Token #{INFLUX_TOKEN}",
          "--header", "Content-Type: text/plain; charset=utf-8",
          "--header", "Accept: application/json",
          "--data-binary", "#{measurement},#{tags} #{fields} #{Time.now.to_i}"
        ]

        # Second precision is highest we can do and has the lowest performance cost.
        url = "#{INFLUX_HOST}/api/v2/write?bucket=#{INFLUX_BUCKET}&precision=s"
        deferred_curl(url, args)
      end

      sig { params(url: String, args: T::Array[String]).void }
      def deferred_curl(url, args)
        curl = Utils::Curl.curl_executable
        if ENV["HOMEBREW_ANALYTICS_DEBUG"]
          puts "#{curl} #{args.join(" ")} \"#{url}\""
          puts Utils.popen_read(curl, *args, url)
        else
          pid = fork do
            exec curl, *args, "--silent", "--output", "/dev/null", url
          end
          Process.detach T.must(pid)
        end
      end

      sig {
        params(measurement: Symbol, package_name: String, tap_name: String,
               on_request: T::Boolean, options: String).void
      }
      def report_event(measurement, package_name:, tap_name:, on_request:, options: "")
        report_influx_event(measurement, package_name: package_name, tap_name: tap_name, on_request: on_request,
options: options)

        package_and_options = package_name
        if tap_name.present? && tap_name != "homebrew/core" && tap_name != "homebrew/cask"
          package_and_options = "#{tap_name}/#{package_and_options}"
        end
        package_and_options = "#{package_and_options} #{options}" if options.present?
        report_google_event(measurement, package_and_options, on_request: on_request)
      end

      sig { params(category: Symbol, action: String, on_request: T::Boolean).void }
      def report_google_event(category, action, on_request: false)
        return if not_this_run? || disabled? || Homebrew::EnvConfig.no_google_analytics?

        category = "install" if category == :formula_install

        report_google(:event,
                      ec: category,
                      ea: action,
                      el: label_google,
                      ev: nil)

        return unless on_request

        report_google(:event,
                      ec: :install_on_request,
                      ea: action,
                      el: label_google,
                      ev: nil)
      end

      sig {
        params(measurement: Symbol, package_name: String, tap_name: String, on_request: T::Boolean,
               options: String).void
      }
      def report_influx_event(measurement, package_name:, tap_name:, on_request: false, options: "")
        return if not_this_run? || disabled?

        report_influx(measurement, package_name: package_name, tap_name: tap_name, on_request: on_request,
options: options)
      end

      sig { params(exception: BuildError).void }
      def report_build_error(exception)
        report_google_build_error(exception)
        report_influx_error(exception)
      end

      sig { params(exception: BuildError).void }
      def report_google_build_error(exception)
        return if not_this_run? || disabled?

        return unless exception.formula.tap
        return unless exception.formula.tap.should_report_analytics?

        formula_full_name = exception.formula.full_name
        package_and_options = if (options = exception.options.to_a.map(&:to_s).join(" ").presence)
          "#{formula_full_name} #{options}".strip
        else
          formula_full_name
        end
        report_google_event(:BuildError, package_and_options)
      end

      sig { params(exception: BuildError).void }
      def report_influx_error(exception)
        return if not_this_run? || disabled?

        formula = exception.formula
        return unless formula

        tap = formula.tap
        return unless tap
        return unless tap.should_report_analytics?

        options = exception.options.to_a.map(&:to_s).join(" ")
        report_influx_event(:build_error, package_name: formula.name, tap_name: tap.name, options: options)
      end

      def messages_displayed?
        config_true?(:analyticsmessage) && config_true?(:caskanalyticsmessage)
      end

      def disabled?
        return true if Homebrew::EnvConfig.no_analytics?

        config_true?(:analyticsdisabled)
      end

      def not_this_run?
        ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"].present?
      end

      def no_message_output?
        # Used by Homebrew/install
        ENV["HOMEBREW_NO_ANALYTICS_MESSAGE_OUTPUT"].present?
      end

      def messages_displayed!
        Homebrew::Settings.write :analyticsmessage, true
        Homebrew::Settings.write :caskanalyticsmessage, true
      end

      def enable!
        Homebrew::Settings.write :analyticsdisabled, false
        delete_uuid!
        messages_displayed!
      end

      def disable!
        Homebrew::Settings.write :analyticsdisabled, true
        delete_uuid!
      end

      def delete_uuid!
        Homebrew::Settings.delete :analyticsuuid
      end

      def output(args:, filter: nil)
        days = args.days || "30"
        category = args.category || "install"
        begin
          json = Homebrew::API::Analytics.fetch category, days
        rescue ArgumentError
          # Ignore failed API requests
          return
        end
        return if json.blank? || json["items"].blank?

        os_version = category == "os-version"
        cask_install = category == "cask-install"
        results = {}
        json["items"].each do |item|
          key = if os_version
            item["os_version"]
          elsif cask_install
            item["cask"]
          else
            item["formula"]
          end
          next if filter.present? && key != filter && !key.start_with?("#{filter} ")

          results[key] = item["count"].tr(",", "").to_i
        end

        if filter.present? && results.blank?
          onoe "No results matching `#{filter}` found!"
          return
        end

        table_output(category, days, results, os_version: os_version, cask_install: cask_install)
      end

      def get_analytics(json, args:)
        full_analytics = args.analytics? || verbose?

        ohai "Analytics"
        json["analytics"].each do |category, value|
          category = category.tr("_", "-")
          analytics = []

          value.each do |days, results|
            days = days.to_i
            if full_analytics
              next if args.days.present? && args.days&.to_i != days
              next if args.category.present? && args.category != category

              table_output(category, days, results)
            else
              total_count = results.values.inject("+")
              analytics << "#{number_readable(total_count)} (#{days} days)"
            end
          end

          puts "#{category}: #{analytics.join(", ")}" unless full_analytics
        end
      end

      def formula_output(formula, args:)
        return if Homebrew::EnvConfig.no_analytics? || Homebrew::EnvConfig.no_github_api?

        json = Homebrew::API::Formula.fetch formula.name
        return if json.blank? || json["analytics"].blank?

        get_analytics(json, args: args)
      rescue ArgumentError
        # Ignore failed API requests
        nil
      end

      def cask_output(cask, args:)
        return if Homebrew::EnvConfig.no_analytics? || Homebrew::EnvConfig.no_github_api?

        json = Homebrew::API::Cask.fetch cask.token
        return if json.blank? || json["analytics"].blank?

        get_analytics(json, args: args)
      rescue ArgumentError
        # Ignore failed API requests
        nil
      end

      sig { returns(String) }
      def custom_prefix_label_google
        "custom-prefix"
      end
      alias generic_custom_prefix_label_google custom_prefix_label_google

      sig { returns(String) }
      def arch_label_google
        if Hardware::CPU.arm?
          "ARM"
        else
          ""
        end
      end
      alias generic_arch_label_google arch_label_google

      def clear_cache
        remove_instance_variable(:@label_google) if instance_variable_defined?(:@label_google)
        remove_instance_variable(:@default_tags_influx) if instance_variable_defined?(:@default_tags_influx)
        remove_instance_variable(:@default_fields_influx) if instance_variable_defined?(:@default_fields_influx)
      end

      sig { returns(String) }
      def label_google
        @label_google ||= begin
          os = OS_VERSION
          arch = ", #{arch_label_google}" if arch_label_google.present?
          prefix = ", #{custom_prefix_label_google}" unless Homebrew.default_prefix?
          ci = ", CI" if ENV["CI"]
          "#{os}#{arch}#{prefix}#{ci}"
        end
      end

      sig { returns(T::Hash[Symbol, String]) }
      def default_tags_influx
        @default_tags_influx ||= begin
          # Only display default prefixes to reduce cardinality and improve privacy
          prefix = Homebrew.default_prefix? ? HOMEBREW_PREFIX.to_s : "custom-prefix"

          # Tags are always strings and must have low cardinality.
          {
            ci:             ENV["CI"].present?,
            prefix:         prefix,
            default_prefix: Homebrew.default_prefix?,
            developer:      Homebrew::EnvConfig.developer?,
            devcmdrun:      config_true?(:devcmdrun),
            arch:           HOMEBREW_PHYSICAL_PROCESSOR,
            os:             HOMEBREW_SYSTEM,
          }
        end
      end

      # remove os_version starting with " or number
      # remove macOS patch release
      sig { returns(T::Hash[Symbol, String]) }
      def default_fields_influx
        @default_fields_influx ||= begin
          version = HOMEBREW_VERSION.match(/^[\d.]+/)[0]
          version = "#{version}-dev" if HOMEBREW_VERSION.include?("-")

          # Only include OS versions with an actual name.
          os_name_and_version = if (os_version = OS_VERSION.presence) && os_version.downcase.match?(/^[a-z]/)
            os_version
          end

          {
            version:             version,
            os_name_and_version: os_name_and_version,
          }
        end
      end

      def table_output(category, days, results, os_version: false, cask_install: false)
        oh1 "#{category} (#{days} days)"
        total_count = results.values.inject("+")
        formatted_total_count = format_count(total_count)
        formatted_total_percent = format_percent(100)

        index_header = "Index"
        count_header = "Count"
        percent_header = "Percent"
        name_with_options_header = if os_version
          "macOS Version"
        elsif cask_install
          "Token"
        else
          "Name (with options)"
        end

        total_index_footer = "Total"
        max_index_width = results.length.to_s.length
        index_width = [
          index_header.length,
          total_index_footer.length,
          max_index_width,
        ].max
        count_width = [
          count_header.length,
          formatted_total_count.length,
        ].max
        percent_width = [
          percent_header.length,
          formatted_total_percent.length,
        ].max
        name_with_options_width = Tty.width -
                                  index_width -
                                  count_width -
                                  percent_width -
                                  10 # spacing and lines

        formatted_index_header =
          format "%#{index_width}s", index_header
        formatted_name_with_options_header =
          format "%-#{name_with_options_width}s",
                 name_with_options_header[0..name_with_options_width-1]
        formatted_count_header =
          format "%#{count_width}s", count_header
        formatted_percent_header =
          format "%#{percent_width}s", percent_header
        puts "#{formatted_index_header} | #{formatted_name_with_options_header} | " \
             "#{formatted_count_header} |  #{formatted_percent_header}"

        columns_line = "#{"-"*index_width}:|-#{"-"*name_with_options_width}-|-" \
                       "#{"-"*count_width}:|-#{"-"*percent_width}:"
        puts columns_line

        index = 0
        results.each do |name_with_options, count|
          index += 1
          formatted_index = format "%0#{max_index_width}d", index
          formatted_index = format "%-#{index_width}s", formatted_index
          formatted_name_with_options =
            format "%-#{name_with_options_width}s",
                   name_with_options[0..name_with_options_width-1]
          formatted_count = format "%#{count_width}s", format_count(count)
          formatted_percent = if total_count.zero?
            format "%#{percent_width}s", format_percent(0)
          else
            format "%#{percent_width}s",
                   format_percent((count.to_i * 100) / total_count.to_f)
          end
          puts "#{formatted_index} | #{formatted_name_with_options} | " \
               "#{formatted_count} | #{formatted_percent}%"
          next if index > 10
        end
        return unless results.length > 1

        formatted_total_footer =
          format "%-#{index_width}s", total_index_footer
        formatted_blank_footer =
          format "%-#{name_with_options_width}s", ""
        formatted_total_count_footer =
          format "%#{count_width}s", formatted_total_count
        formatted_total_percent_footer =
          format "%#{percent_width}s", formatted_total_percent
        puts "#{formatted_total_footer} | #{formatted_blank_footer} | " \
             "#{formatted_total_count_footer} | #{formatted_total_percent_footer}%"
      end

      def config_true?(key)
        Homebrew::Settings.read(key) == "true"
      end

      def format_count(count)
        count.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
      end

      def format_percent(percent)
        format("%<percent>.2f", percent: percent)
      end
    end
  end
end

require "extend/os/utils/analytics"
