# frozen_string_literal: true

require "erb"

module Utils
  module Analytics
    class << self
      def report(type, metadata = {})
        return if disabled?

        args = []

        # do not load .curlrc unless requested (must be the first argument)
        args << "-q" unless ENV["HOMEBREW_CURLRC"]

        args += %W[
          --max-time 3
          --user-agent #{HOMEBREW_USER_AGENT_CURL}
          --data v=1
          --data aip=1
          --data t=#{type}
          --data tid=#{ENV["HOMEBREW_ANALYTICS_ID"]}
          --data cid=#{ENV["HOMEBREW_ANALYTICS_USER_UUID"]}
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

        # Send analytics. Don't send or store any personally identifiable information.
        # https://docs.brew.sh/Analytics
        # https://developers.google.com/analytics/devguides/collection/protocol/v1/devguide
        # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters
        if ENV["HOMEBREW_ANALYTICS_DEBUG"]
          url = "https://www.google-analytics.com/debug/collect"
          puts "#{ENV["HOMEBREW_CURL"]} #{args.join(" ")} #{url}"
          puts Utils.popen_read ENV["HOMEBREW_CURL"], *args, url
        else
          pid = fork do
            exec ENV["HOMEBREW_CURL"],
                 *args,
                 "--silent", "--output", "/dev/null",
                 "https://www.google-analytics.com/collect"
          end
          Process.detach pid
        end
      end

      def report_event(category, action, label = os_prefix_ci, value = nil)
        report(:event,
               ec: category,
               ea: action,
               el: label,
               ev: value)
      end

      def report_build_error(exception)
        return unless exception.formula.tap
        return unless exception.formula.tap.installed?
        return if exception.formula.tap.private?

        action = exception.formula.full_name
        if (options = exception.options&.to_a&.join(" "))
          action = "#{action} #{options}".strip
        end
        report_event("BuildError", action)
      end

      def messages_displayed?
        config_true?(:analyticsmessage) && config_true?(:caskanalyticsmessage)
      end

      def disabled?
        return true if ENV["HOMEBREW_NO_ANALYTICS"] || ENV["HOMEBREW_NO_ANALYTICS_THIS_RUN"]

        config_true?(:analyticsdisabled)
      end

      def no_message_output?
        # Used by Homebrew/install
        ENV["HOMEBREW_NO_ANALYTICS_MESSAGE_OUTPUT"].present?
      end

      def uuid
        config_get(:analyticsuuid)
      end

      def messages_displayed!
        config_set(:analyticsmessage, true)
        config_set(:caskanalyticsmessage, true)
      end

      def enable!
        config_set(:analyticsdisabled, false)
        messages_displayed!
      end

      def disable!
        config_set(:analyticsdisabled, true)
        regenerate_uuid!
      end

      def regenerate_uuid!
        # it will be regenerated in next run unless disabled.
        config_delete(:analyticsuuid)
      end

      def output(filter: nil)
        days = Homebrew.args.days || "30"
        category = Homebrew.args.category || "install"
        json = formulae_api_json("analytics/#{category}/#{days}d.json")
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
          if filter.present?
            next if key != filter && !key.start_with?("#{filter} ")
          end
          results[key] = item["count"].tr(",", "").to_i
        end

        if filter.present? && results.blank?
          onoe "No results matching `#{filter}` found!"
          return
        end

        table_output(category, days, results, os_version: os_version, cask_install: cask_install)
      end

      def formula_output(f)
        json = formulae_api_json("#{formula_path}/#{f}.json")
        return if json.blank? || json["analytics"].blank?

        full_analytics = Homebrew.args.analytics? || Homebrew.args.verbose?

        ohai "Analytics"
        json["analytics"].each do |category, value|
          category = category.tr("_", "-")
          analytics = []

          value.each do |days, results|
            days = days.to_i
            if full_analytics
              if Homebrew.args.days.present?
                next if Homebrew.args.days&.to_i != days
              end
              if Homebrew.args.category.present?
                next if Homebrew.args.category != category
              end

              table_output(category, days, results)
            else
              total_count = results.values.inject("+")
              analytics << "#{number_readable(total_count)} (#{days} days)"
            end
          end

          puts "#{category}: #{analytics.join(", ")}" unless full_analytics
        end
      end

      def custom_prefix_label
        "custom-prefix"
      end

      def clear_os_prefix_ci
        return unless instance_variable_defined?(:@os_prefix_ci)

        remove_instance_variable(:@os_prefix_ci)
      end

      def os_prefix_ci
        @os_prefix_ci ||= begin
          os = OS_VERSION
          prefix = ", #{custom_prefix_label}" unless Homebrew.default_prefix?
          ci = ", CI" if ENV["CI"]
          "#{os}#{prefix}#{ci}"
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
        puts "#{formatted_index_header} | #{formatted_name_with_options_header} | "\
            "#{formatted_count_header} |  #{formatted_percent_header}"

        columns_line = "#{"-"*index_width}:|-#{"-"*name_with_options_width}-|-"\
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
        puts "#{formatted_total_footer} | #{formatted_blank_footer} | "\
            "#{formatted_total_count_footer} | #{formatted_total_percent_footer}%"
      end

      def config_true?(key)
        config_get(key) == "true"
      end

      def config_get(key)
        HOMEBREW_REPOSITORY.cd do
          Utils.popen_read("git", "config", "--get", "homebrew.#{key}").chomp
        end
      end

      def config_set(key, value)
        HOMEBREW_REPOSITORY.cd do
          safe_system "git", "config", "--replace-all", "homebrew.#{key}", value.to_s
        end
      end

      def config_delete(key)
        HOMEBREW_REPOSITORY.cd do
          system "git", "config", "--unset-all", "homebrew.#{key}"
        end
      end

      def formulae_api_json(endpoint)
        return if ENV["HOMEBREW_NO_ANALYTICS"] || ENV["HOMEBREW_NO_GITHUB_API"]

        output, = curl_output("--max-time", "5",
                              "https://formulae.brew.sh/api/#{endpoint}")
        return if output.blank?

        JSON.parse(output)
      rescue JSON::ParserError
        nil
      end

      def format_count(count)
        count.to_s.reverse.gsub(/(\d{3})(?=\d)/, '\\1,').reverse
      end

      def format_percent(percent)
        format("%<percent>.2f", percent: percent)
      end

      def formula_path
        "formula"
      end
      alias generic_formula_path formula_path

      def analytics_path
        "analytics"
      end
      alias generic_analytics_path analytics_path
    end
  end
end

require "extend/os/utils/analytics"
