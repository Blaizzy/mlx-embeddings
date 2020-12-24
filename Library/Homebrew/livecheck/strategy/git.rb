# typed: false
# frozen_string_literal: true

require "open3"

module Homebrew
  module Livecheck
    module Strategy
      # The {Git} strategy identifies versions of software in a Git repository
      # by checking the tags using `git ls-remote --tags`.
      #
      # Livecheck has historically prioritized the {Git} strategy over others
      # and this behavior was continued when the priority setup was created.
      # This is partly related to Livecheck checking formula URLs in order of
      # `head`, `stable`, and then `homepage`. The higher priority here may
      # be removed (or altered) in the future if we reevaluate this particular
      # behavior.
      #
      # This strategy does not have a default regex. Instead, it simply removes
      # any non-digit text from the start of tags and parses the rest as a
      # {Version}. This works for some simple situations but even one unusual
      # tag can cause a bad result. It's better to provide a regex in a
      # `livecheck` block, so `livecheck` only matches what we really want.
      #
      # @api public
      class Git
        # The priority of the strategy on an informal scale of 1 to 10 (from
        # lowest to highest).
        PRIORITY = 8

        # Fetches a remote Git repository's tags using `git ls-remote --tags`
        # and parses the command's output. If a regex is provided, it will be
        # used to filter out any tags that don't match it.
        #
        # @param url [String] the URL of the Git repository to check
        # @param regex [Regexp] the regex to use for filtering tags
        # @return [Hash]
        def self.tag_info(url, regex = nil)
          # Open3#capture3 is used here because we need to capture stderr
          # output and handle it in an appropriate manner. Alternatives like
          # SystemCommand always print errors (as well as debug output) and
          # don't meet the same goals.
          stdout_str, stderr_str, _status = Open3.capture3(
            { "GIT_TERMINAL_PROMPT" => "0" }, "git", "ls-remote", "--tags", url
          )

          tags_data = { tags: [] }
          tags_data[:messages] = stderr_str.split("\n") if stderr_str.present?
          return tags_data if stdout_str.blank?

          # Isolate tag strings by removing leading/trailing text
          stdout_str.gsub!(%r{^.*\trefs/tags/}, "")
          stdout_str.gsub!("^{}", "")

          tags = stdout_str.split("\n").uniq.sort
          tags.select! { |t| t =~ regex } if regex
          tags_data[:tags] = tags

          tags_data
        end

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        def self.match?(url)
          (DownloadStrategyDetector.detect(url) <= GitDownloadStrategy) == true
        end

        # Checks the Git tags for new versions. When a regex isn't provided,
        # this strategy simply removes non-digits from the start of tag
        # strings and parses the remaining text as a {Version}.
        #
        # @param url [String] the URL of the Git repository to check
        # @param regex [Regexp] the regex to use for matching versions
        # @return [Hash]
        def self.find_versions(url, regex = nil, &block)
          match_data = { matches: {}, regex: regex, url: url }

          tags_data = tag_info(url, regex)

          if tags_data.key?(:messages)
            match_data[:messages] = tags_data[:messages]
            return match_data if tags_data[:tags].blank?
          end

          tags_only_debian = tags_data[:tags].all? { |tag| tag.start_with?("debian/") }

          if block
            case (value = block.call(tags_data[:tags], regex))
            when String
              match_data[:matches][value] = Version.new(value)
            when Array
              value.each do |tag|
                match_data[:matches][tag] = Version.new(tag)
              end
            else
              raise TypeError, "Return value of `strategy :git` block must be a string or array of strings."
            end

            return match_data
          end

          tags_data[:tags].each do |tag|
            # Skip tag if it has a 'debian/' prefix and upstream does not do
            # only 'debian/' prefixed tags
            next if tag =~ %r{^debian/} && !tags_only_debian

            captures = regex.is_a?(Regexp) ? tag.scan(regex) : []
            tag_cleaned = if captures[0].is_a?(Array)
              captures[0][0] # Use the first capture group (the version)
            else
              tag[/\D*(.*)/, 1] # Remove non-digits from the start of the tag
            end

            match_data[:matches][tag] = Version.new(tag_cleaned)
          rescue TypeError
            nil
          end

          match_data
        end
      end
    end
  end
end
