require "net/http"
require "open-uri"

class HomebrewFormula

  def generate_new_download_url(outdated_url, old_version, latest_version)
      if [outdated_url, old_version, latest_version].include? nil
        puts "\n- Could not generate download url"
        nil
      else
        puts "\n- Generating download url"
        outdated_url.gsub(old_version, latest_version)
      end
  end

  def generate_checksum(new_url)
    begin
      puts "- Generating checksum for url: #{new_url}"
      tempfile = URI.parse(new_url).open
      tempfile.close
      return Digest::SHA256.file(tempfile.path).hexdigest
    rescue
      puts "- Failed to generate Checksum \n"
      return nil
    end
  end
end
