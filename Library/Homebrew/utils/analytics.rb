# typed: false
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
    class << self
      extend T::Sig

      include Context

      INFLUX_BUCKET = "analytics"
      INFLUX_TOKEN = "y2JZsgE7glWT9V-S-nElETLp8oyH9PGh9JVa-kCdOdp7mEHIOws4BtdjsIe3HHpCBty7IQHLnmh0prqK2vBj9A=="
      INFLUX_HOST = "europe-west1-1.gcp.cloud2.influxdata.com"

      sig { params(type: T.any(String, Symbol), metadata: T::Hash[Symbol, T.untyped]).void }
      def report_google(type, metadata = {})
        os = metadata[:el][:os]
        arch = ", #{metadata[:el][:arch]}" if metadata[:el][:arch].present?
        prefix = ", #{metadata[:el][:prefix]}" if metadata[:el][:prefix].present?
        ci = ", CI" if metadata[:el][:CI] == true

        metadata[:el] = "#{os}#{arch}#{prefix}#{ci}"

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
            --data cid=#{ENV.fetch("HOMEBREW_ANALYTICS_USER_UUID")}
            --data an=#{HOMEBREW_PRODUCT}
            --data av=#{HOMEBREW_VERSION}
          ]
          metadata.each do |key, value|
            next unless key
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
                   "https://www.google-analytqics.com/collect"
            end
            Process.detach T.must(pid)
          end
        end
        nil
      end

      sig {
        params(category: T.any(String, Symbol), action: T.any(String, Symbol), on_request: T::Boolean,
               additional_tags: T::Hash[Symbol, T.untyped]).void
      }
      def report_influx(category, action, on_request, additional_tags = {})
        return unless ENV["HOMEBREW_ANALYTICS_ENABLE_INFLUX"]

        # Append general information to device information
        tags = additional_tags.merge(action: action, on_request: !on_request.nil?)
                              .compact_blank
                              .map { |k, v| "#{k}=#{v.to_s.sub(" ", "\\ ")}" } # convert to key/value parameters
                              .join(",")

        args = [
          "--max-time", "3",
          "--header", "Content-Type: text/plain; charset=utf-8",
          "--header", "Accept: application/json",
          "--header", "Authorization: Token #{INFLUX_TOKEN}",
          "--data-raw", "#{category},#{tags} count=1i #{Time.now.to_i}"
        ]

        curl = Utils::Curl.curl_executable
        url = "https://#{INFLUX_HOST}/api/v2/write?bucket=#{INFLUX_BUCKET}&precision=s"
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

      sig { params(category: T.any(String, Symbol), action: String, on_request: T::Boolean).void }
      def report_event(category, action, on_request: false)
        return if not_this_run?
        return if disabled?

        google_label = os_arch_prefix_ci(verbose: false)

        report_google(:event,
                      ec: category,
                      ea: action,
                      el: google_label,
                      ev: nil)

        if on_request
          report_google(:event,
                        ec: :install_on_request,
                        ea: action,
                        el: google_label,
                        ev: nil)
        end

        influx_additional_data = os_arch_prefix_ci(verbose: true)
        report_influx(category, action, on_request, influx_additional_data)
      end

      def report_build_error(exception)
        return unless exception.formula.tap
        return unless exception.formula.tap.installed?
        return if exception.formula.tap.private?

        action = exception.formula.full_name
        if (options = exception.options.to_a.map(&:to_s).join(" ").presence)
          action = "#{action} #{options}".strip
        end
        report_event("BuildError", action)
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

      def uuid
        Homebrew::Settings.read :analyticsuuid
      end

      def messages_displayed!
        Homebrew::Settings.write :analyticsmessage, true
        Homebrew::Settings.write :caskanalyticsmessage, true
      end

      def enable!
        Homebrew::Settings.write :analyticsdisabled, false
        messages_displayed!
      end

      def disable!
        Homebrew::Settings.write :analyticsdisabled, true
        regenerate_uuid!
      end

      def regenerate_uuid!
        # it will be regenerated in next run unless disabled.
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

      def formula_output(f, args:)
        return if Homebrew::EnvConfig.no_analytics? || Homebrew::EnvConfig.no_github_api?

        json = Homebrew::API::Formula.fetch f.name
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

      sig { params(verbose: T::Boolean).returns(String) }
      def custom_prefix_label(verbose: false)
        "custom-prefix"
      end
      alias generic_custom_prefix_label custom_prefix_label

      sig { params(verbose: T::Boolean).returns(String) }
      def arch_label(verbose: false)
        if Hardware::CPU.arm?
          "ARM"
        else
          ""
        end
      end

      def clear_os_arch_prefix_ci
        return unless instance_variable_defined?(:@os_arch_prefix_ci)

        remove_instance_variable(:@os_arch_prefix_ci)
      end

      sig { params(verbose: T::Boolean).returns(T::Hash[Symbol, String]) }
      def os_arch_prefix_ci(verbose: false)
        @os_arch_prefix_ci ||= begin
          data = {
            os:        OS_VERSION,
            developer: Homebrew::EnvConfig.developer?,
            version:   HOMEBREW_VERSION,
            system:    HOMEBREW_SYSTEM,
            ci:        ENV["CI"].present?,
            arch:      arch_label(verbose: verbose),
            prefix:    custom_prefix_label(verbose: verbose),
          }
          unless verbose
            data.delete(:arch) if data[:arch].blank?
            data.delete(:prefix) if Homebrew.default_prefix?
          end

          data
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
