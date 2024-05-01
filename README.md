# Customer Segmentation

## Overview
This project aims to segment customers based on their purchasing behavior using machine learning techniques. By identifying different customer segments, businesses can tailor their marketing strategies, product offerings, and customer service to better meet the needs of each segment.

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/sofiacan97/customer_segmentation.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Approach
1. **Data Preprocessing**: Clean the data, handle missing values, and normalize the features if necessary.
2. **Feature Selection**: Identify relevant features for customer segmentation.
3. **Model Training**: Apply machine learning algorithms such as K-means clustering or KMedoids clustering to segment the customers.
4. **Evaluation**: Evaluate the quality of the clustering results using metrics like silhouette score or inertia.
5. **Interpretation**: Analyze the characteristics of each customer segment and derive actionable insights for business decisions.

## Notes
- This project does not consider a starting dataset.
- To ensure compatibility with the code provided in this project, make sure your dataset adheres to the following specifications:

### Dataset Specifications:
- **Format**: The dataset should be structured, ideally in a CSV (Comma Separated Values) file or another tabular format supported by pandas DataFrame.
- **Columns**: Your dataset should include the following columns:
    - CustomerID: Unique identifier for each customer.
    - Company: Company or organization associated with the customer.
    - Age: Age of the customer.
    - Language: Language preference of the customer.
    - Tenure: Length of time the customer has been with the company.
    - Loyalty: Loyalty status of the customer.
    - Tier: Customer tier or level within the loyalty program.
    - Client Type: Type of client (e.g., Retail, Wholesale).
    - Frequency: Frequency of customer purchases.
    - AOV: Average Order Value (AOV) representing the average amount spent per transaction.
    - TopStoreSite: Preferred store or site frequented by the customer.
    - TopFamily: Preferred product family or category.
    - TopWeekday: Preferred weekday for making purchases.
By preparing your dataset according to these guidelines, you can effectively utilize the provided code for customer segmentation analysis. For additional assistance or clarification, refer to the project documentation or reach out to the project owner.
